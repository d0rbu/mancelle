"""Tests for the Pass-1 sorted-magnitude spill precision contract.

Audit: phase1/08-compute-statistics.md + phase2/03-numerical-correctness.md §1 P0 #1.

History: an early version spilled sorted magnitudes via
`t.save(sorted_mags.to(t.float16).contiguous(), ...)`. Any magnitude above
fp16's finite range (±65504) became +Inf, and sub-fp16-precision values were
rounded — silently corrupting downstream percentiles. The fix made the spill
unconditionally fp32 (2x disk).

Refined contract (this module): `_spill_sorted_mags` spills in the **smallest
precision that is bitwise-lossless for the given tensor** — fp16 iff
`fp32 -> fp16 -> fp32` is the exact identity for *every* element, else fp32.

- In the production pipeline `feature_mags` are stored fp16 in shards
  (`feature_io._optimize_dtype` raises rather than ever writing fp32;
  `FP16_SAFE_MAX=60000` guard), upcast fp16->fp32 on load, then only
  *permuted* by the sort (no arithmetic). So every spilled value is exactly
  fp16-representable and the fp16 spill is provably lossless at half the
  Pass-1 scratch cost.
- Any out-of-fp16-range input (>65504 -> +Inf, or sub-fp16 precision /
  subnormal) is *not* bitwise-lossless, so it still falls back to fp32 —
  preserving the original audit guarantee. `TestSpillDtypeAndFiniteness` /
  `TestPercentilesMatchFP64Oracle` pin that fallback; `TestLosslessFp16Spill`
  pins the new fp16-when-lossless behaviour and that Pass-2 percentiles are
  bitwise-identical to the fp32 path.

`_read_feature_mags` already upcasts whatever dtype it reads to fp32, so the
Pass-2 read path is unchanged regardless of spill precision.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch as t

from steering.compute_statistics import compute_per_feature_percentiles
from steering.feature_io import load_features, save_features
from tests.steering._compute_statistics_reference import (
    SortedShardInfo,
    _can_spill_as_fp16_losslessly,
    _read_feature_mags,
    _setup_merge_context,
    _spill_sorted_mags,
    merge_per_feature_percentiles,
    optimized_pass1,
)

# Magnitudes that exceed fp16's max finite value (65504).
# Also include a tiny value near fp16's smallest normal (~6.1e-5) for the lower tail.
EXTREME_LARGE = [70000.0, 80000.0, 100000.0, 200000.0, 500000.0, 1e6]
EXTREME_SMALL = [1e-8, 1e-7]
NORMAL_MAGS = [0.1, 1.0, 5.0, 42.0, 500.0, 1000.0, 5000.0, 20000.0, 60000.0]


def _spill(mags: t.Tensor, path: Path) -> None:
    """Mimic the production spill helper: write sorted mags to disk."""
    _spill_sorted_mags(mags, path)


class TestSpillDtypeAndFiniteness:
    def test_spilled_tensor_is_float32(self):
        """The spilled tensor must be loaded back as float32, not float16."""
        mags = t.tensor([1.0, 2.0, 70000.0, 1e-8], dtype=t.float32).sort().values
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mags.pt"
            _spill(mags, path)
            loaded = t.load(path, weights_only=True)

        assert loaded.dtype == t.float32, (
            f"Spilled dtype is {loaded.dtype}; must be float32 so "
            f"magnitudes above 65504 are preserved."
        )

    def test_no_inf_after_spill_with_extreme_values(self):
        """Extreme values > 65504 must survive the round-trip without becoming Inf."""
        mags = t.tensor(
            NORMAL_MAGS + EXTREME_LARGE + EXTREME_SMALL, dtype=t.float32
        ).sort().values

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mags.pt"
            _spill(mags, path)
            loaded = t.load(path, weights_only=True)

        assert t.isfinite(loaded).all(), (
            f"Spilled tensor has non-finite values: "
            f"{loaded[~t.isfinite(loaded)].tolist()}"
        )
        # Values must match the input exactly (fp32 preserves all these).
        assert t.allclose(loaded.to(t.float64), mags.to(t.float64), rtol=0, atol=0), (
            f"Spilled values drifted: max abs diff "
            f"{(loaded.to(t.float64) - mags.to(t.float64)).abs().max().item()}"
        )


class TestPercentilesMatchFP64Oracle:
    """The critical test: percentiles from the spill path must match a
    fp64 reference computed directly from the original magnitudes, even
    when the magnitudes exceed fp16's representable range.
    """

    def _build_sorted_shard(
        self, tmp: Path, feature_id: int, mags: t.Tensor
    ) -> tuple[SortedShardInfo, int]:
        """Construct a one-feature sorted shard and spill its mags."""
        sorted_mags = mags.sort().values
        n_act = len(sorted_mags)
        max_fid = feature_id
        per_feature_counts = t.zeros(max_fid + 1, dtype=t.int64)
        per_feature_counts[feature_id] = n_act

        path = tmp / f"shard_{feature_id:05d}_mags.pt"
        _spill(sorted_mags, path)

        return (
            SortedShardInfo(
                mags_path=path,
                n_activations=n_act,
                per_feature_counts=per_feature_counts,
            ),
            n_act,
        )

    def test_single_feature_extreme_mags_match_oracle(self):
        """Feature with magnitudes > 65504: computed percentiles must match
        a fp64 torch.quantile oracle within float32 tolerance.
        """
        feature_id = 0
        # Mix of normal and extreme values; make enough points for meaningful pctiles.
        # Use a wide range spanning fp16-unsafe territory.
        t.manual_seed(42)
        base = t.rand(5000, dtype=t.float64) * 1000.0  # normal range
        extreme = t.tensor(EXTREME_LARGE, dtype=t.float64).repeat(50)  # 300 extreme
        all_mags_f64 = t.cat([base, extreme])
        all_mags = all_mags_f64.to(t.float32)

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            shard, n_act = self._build_sorted_shard(tmp, feature_id, all_mags)

            n_sae_features = feature_id + 1
            occurrence = t.zeros(n_sae_features, dtype=t.int64)
            occurrence[feature_id] = n_act
            mag_subsample = all_mags.clone()
            global_mag_min = float(all_mags.min().item())

            result, _bin_edges = merge_per_feature_percentiles(
                [shard],
                n_sae_features=n_sae_features,
                occurrence=occurrence,
                mag_subsample=mag_subsample,
                global_mag_min=global_mag_min,
                percentiles=(50, 90, 99),
            )

        assert feature_id in result, "Feature missing from percentile result"
        pcts = result[feature_id].percentiles

        # fp64 oracle via torch.quantile.
        sorted_f64 = all_mags_f64.sort().values
        for p in (50, 90, 99):
            # Linear interpolation on sorted fp64 data.
            idx = (len(sorted_f64) - 1) * (p / 100.0)
            lo = int(idx)
            hi = min(lo + 1, len(sorted_f64) - 1)
            frac = idx - lo
            oracle = sorted_f64[lo].item() + frac * (
                sorted_f64[hi].item() - sorted_f64[lo].item()
            )
            got = pcts[f"p{p}"]

            assert t.isfinite(t.tensor(got)), (
                f"p{p} is non-finite: {got} — spill overflow?"
            )
            # Tolerance: float32 relative precision is ~6e-8; computed percentile
            # was rounded to 6 decimal places, giving ~1e-6 absolute at scale 1.
            # For values up to 1e6, allow generous rtol + atol.
            abs_err = abs(got - oracle)
            rel_err = abs_err / max(abs(oracle), 1.0)
            assert rel_err < 1e-4 and abs_err < 1.0, (
                f"p{p} mismatch: got {got}, oracle {oracle}, "
                f"abs_err {abs_err:.6g}, rel_err {rel_err:.6g}. "
                f"Likely fp16 spill overflow corrupted tail values."
            )

    def test_maximum_is_preserved(self):
        """Specifically: the maximum (p100 ≈ last sorted value) must not
        be truncated to fp16's max (65504)."""
        feature_id = 0
        # One dominant extreme value.
        all_mags = t.tensor([1.0, 10.0, 100.0, 1000.0, 123456.0], dtype=t.float32)

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            shard, n_act = self._build_sorted_shard(tmp, feature_id, all_mags)

            n_sae_features = feature_id + 1
            occurrence = t.zeros(n_sae_features, dtype=t.int64)
            occurrence[feature_id] = n_act
            mag_subsample = all_mags.clone()
            global_mag_min = float(all_mags.min().item())

            # Read via merge helpers directly to inspect the raw read path.
            active_ids, shard_offsets, shard_mmaps, _, _, _ = _setup_merge_context(
                [shard],
                n_sae_features,
                occurrence,
                mag_subsample,
                global_mag_min,
            )
            mags_read = _read_feature_mags(feature_id, shard_offsets, shard_mmaps)

        assert mags_read is not None
        assert t.isfinite(mags_read).all(), (
            f"Read mags contain non-finite: {mags_read.tolist()}"
        )
        max_read = float(mags_read.max().item())
        assert max_read == pytest.approx(123456.0, rel=1e-5), (
            f"Maximum was truncated/corrupted: read {max_read}, expected 123456.0. "
            f"fp16 would have saturated this to +Inf."
        )


def _fp16_exact(values: list[float]) -> t.Tensor:
    """fp32 tensor whose every element is exactly fp16-representable.

    Mirrors the production invariant: shard ``feature_mags`` are stored
    fp16 then upcast to fp32 on load, so every value round-trips fp16
    exactly. (Does NOT use ``save_features``; constructs the post-load
    state directly.)
    """
    return t.tensor(values, dtype=t.float32).to(t.float16).to(t.float32)


def _load(path: Path) -> t.Tensor:
    return t.load(path, weights_only=True)


def _write_realistic_saved_shards(feature_dir: Path) -> list[Path]:
    """Write shards through save_features so mags take the real fp16 path."""
    written: list[Path] = []
    for shard_idx in range(2):
        rows = [
            [(2, 0.10000001 + shard_idx), (0, 1.3333333), (1, 57.125)],
            [(1, 2048.25), (2, 3.1415927 + shard_idx), (0, 99.9375)],
            [(0, 0.33333334), (1, 4095.5 + shard_idx), (2, 123.4567)],
        ]
        feat_ids: list[int] = []
        feat_mags: list[float] = []
        offsets = [0]
        for row in rows:
            for fid, mag in row:
                feat_ids.append(fid)
                feat_mags.append(mag)
            offsets.append(len(feat_ids))

        n_segments = len(rows)
        shard = {
            "feature_ids": t.tensor(feat_ids, dtype=t.int32),
            "feature_mags": t.tensor(feat_mags, dtype=t.float32),
            "offsets": t.tensor(offsets, dtype=t.int64),
            "n_segments": n_segments,
            "n_docs": 1,
            "doc_ids": t.zeros(n_segments, dtype=t.int32),
            "doc_token_counts": t.tensor([n_segments * 128], dtype=t.int32),
            "segment_indices": t.arange(n_segments, dtype=t.int32),
            "segment_positions": t.arange(n_segments, dtype=t.int32) * 128,
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
            "context_window": 1024,
            "pooling": "max_pool",
            "layer": 8,
        }
        written.append(
            save_features(shard, feature_dir / f"shard_{shard_idx:05d}_features")
        )
    return written


class TestLosslessFp16Spill:
    """`_spill_sorted_mags` spills fp16 when (and only when) that is a
    bitwise-lossless round-trip, halving Pass-1 scratch for real shards
    without changing any downstream value.
    """

    def test_fp16_representable_data_spills_as_float16(self):
        mags = _fp16_exact([0.1, 1.0, 42.0, 500.0, 1000.0, 20000.0, 59000.0])
        mags = mags.sort().values
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mags.pt"
            _spill(mags, path)
            loaded = _load(path)
        assert loaded.dtype == t.float16, (
            f"fp16-representable data must spill as float16 to halve "
            f"scratch; got {loaded.dtype}"
        )

    def test_fp16_spill_is_bitwise_lossless(self):
        mags = _fp16_exact(
            [5.96e-08, 1e-4, 0.1, 1.0, 42.0, 500.0, 1234.5, 20000.0, 59000.0]
        ).sort().values
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mags.pt"
            _spill(mags, path)
            loaded = _load(path)
        assert t.equal(loaded.to(t.float32), mags), (
            "fp16 spill of fp16-representable data must round-trip "
            "bit-for-bit"
        )

    def test_noncontiguous_fp16_representable_input_spills_losslessly(self):
        """The helper owns contiguity; callers only need to provide values."""
        base = _fp16_exact([0.1, 0.2, 1.0, 2.0, 42.0, 84.0, 500.0, 1000.0])
        mags = base.reshape(4, 2).t().reshape(-1)[::2]
        assert not mags.is_contiguous()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mags.pt"
            _spill(mags, path)
            loaded = _load(path)
        assert loaded.dtype == t.float16
        assert t.equal(loaded.to(t.float32), mags.contiguous())

    @pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
    def test_cuda_input_spills_cpu_backed_tensor(self):
        """Pass-2 mmap expects CPU storage even if a future caller passes CUDA mags."""
        mags = _fp16_exact([1.0, 2.0, 42.0, 500.0]).cuda()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mags.pt"
            _spill(mags, path)
            loaded = _load(path)
        assert loaded.device.type == "cpu"
        assert loaded.dtype == t.float16
        assert t.equal(loaded.to(t.float32), mags.cpu())

    def test_lossless_check_is_chunked_and_requires_finite_values(self, monkeypatch):
        monkeypatch.setattr(
            "tests.steering._compute_statistics_reference.FP16_SPILL_CHECK_CHUNK_SIZE",
            2,
        )
        mags = _fp16_exact([0.1, 1.0, 42.0, 500.0, 9000.0])
        assert _can_spill_as_fp16_losslessly(mags)
        assert not _can_spill_as_fp16_losslessly(
            t.tensor([1.0, float("inf")], dtype=t.float32),
        )
        assert not _can_spill_as_fp16_losslessly(
            t.tensor([0.10000001, 1.0], dtype=t.float32),
        )
        assert not _can_spill_as_fp16_losslessly(
            t.tensor([1.0, 2.0, 0.10000001], dtype=t.float32),
        )

    def test_fp16_spill_roughly_halves_file_size(self):
        mags = _fp16_exact([float(x) for x in range(1, 4097)]).sort().values
        with tempfile.TemporaryDirectory() as tmp:
            p16 = Path(tmp) / "f16.pt"
            p32 = Path(tmp) / "f32.pt"
            _spill(mags, p16)
            t.save(mags.to(t.float32).contiguous(), p32)
            assert _load(p16).dtype == t.float16
            ratio = p16.stat().st_size / p32.stat().st_size
            assert ratio < 0.6, (
                f"fp16 spill should be ~half the fp32 size; ratio={ratio:.3f}"
            )

    def test_single_unsafe_value_forces_fp32_for_whole_tensor(self):
        """One >65504 value among safe ones -> entire spill fp32, exact."""
        mags = t.tensor(
            [1.0, 42.0, 500.0, 70000.0], dtype=t.float32
        ).sort().values
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mags.pt"
            _spill(mags, path)
            loaded = _load(path)
        assert loaded.dtype == t.float32
        assert t.isfinite(loaded).all()
        assert t.equal(loaded, mags)

    def test_nonfinite_value_forces_fp32_for_whole_tensor(self):
        """Inf round-trips through fp16 numerically but is not a safe fp16 spill."""
        mags = t.tensor([1.0, 42.0, float("inf")], dtype=t.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mags.pt"
            _spill(mags, path)
            loaded = _load(path)
        assert loaded.dtype == t.float32
        assert t.equal(loaded, mags)

    def test_nan_value_forces_fp32_for_whole_tensor(self):
        """NaN is not a valid magnitude, but direct callers still get fp32 fallback."""
        mags = t.tensor([1.0, 42.0, float("nan")], dtype=t.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mags.pt"
            _spill(mags, path)
            loaded = _load(path)
        assert loaded.dtype == t.float32
        assert t.isnan(loaded[-1])
        assert t.equal(loaded[:-1], mags[:-1])

    def test_sub_fp16_precision_value_forces_fp32(self):
        """A value with sub-fp16 precision (not exactly fp16) -> fp32."""
        mags = t.tensor(
            [0.10000001, 1.0, 42.0], dtype=t.float32
        ).sort().values
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mags.pt"
            _spill(mags, path)
            loaded = _load(path)
        assert loaded.dtype == t.float32
        assert t.equal(loaded, mags)

    def test_empty_tensor_spill(self):
        mags = t.tensor([], dtype=t.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mags.pt"
            _spill(mags, path)
            loaded = _load(path)
        assert loaded.numel() == 0

    def test_read_path_upcasts_fp16_spill_to_fp32(self):
        """Pass-2 `_read_feature_mags` returns fp32 from an fp16 spill."""
        feature_id = 0
        mags = _fp16_exact([1.0, 7.0, 42.0, 500.0, 9000.0, 59000.0])
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            sorted_mags = mags.sort().values
            n_act = len(sorted_mags)
            per_feature_counts = t.zeros(feature_id + 1, dtype=t.int64)
            per_feature_counts[feature_id] = n_act
            path = tmp / f"shard_{feature_id:05d}_mags.pt"
            _spill(sorted_mags, path)
            assert _load(path).dtype == t.float16, "precondition: fp16 spill"

            shard = SortedShardInfo(
                mags_path=path,
                n_activations=n_act,
                per_feature_counts=per_feature_counts,
            )
            occurrence = t.zeros(feature_id + 1, dtype=t.int64)
            occurrence[feature_id] = n_act
            _ids, shard_offsets, shard_mmaps, _, _, _ = _setup_merge_context(
                [shard],
                feature_id + 1,
                occurrence,
                sorted_mags.clone(),
                float(sorted_mags.min().item()),
            )
            read = _read_feature_mags(feature_id, shard_offsets, shard_mmaps)
        assert read is not None
        assert read.dtype == t.float32
        assert t.equal(read, sorted_mags)

    def test_percentiles_identical_fp16_vs_fp32_spill(self):
        """The core validation: per-feature percentiles from the
        (lossless) fp16 spill are *bitwise identical* to the fp32 spill
        for real-range, fp16-representable data."""
        feature_id = 0
        t.manual_seed(7)
        base = (t.rand(8000, dtype=t.float32) * 50000.0) + 1.0
        mags = base.to(t.float16).to(t.float32).sort().values  # fp16-exact
        n_act = len(mags)
        per_feature_counts = t.zeros(feature_id + 1, dtype=t.int64)
        per_feature_counts[feature_id] = n_act
        occurrence = t.zeros(feature_id + 1, dtype=t.int64)
        occurrence[feature_id] = n_act

        def _percentiles(force_fp32: bool):
            with tempfile.TemporaryDirectory() as tmp_str:
                path = Path(tmp_str) / f"shard_{feature_id:05d}_mags.pt"
                if force_fp32:
                    t.save(mags.to(t.float32).contiguous(), path)
                else:
                    _spill(mags, path)
                    assert _load(path).dtype == t.float16, (
                        "fp16-exact data must take the fp16 path here"
                    )
                shard = SortedShardInfo(
                    mags_path=path,
                    n_activations=n_act,
                    per_feature_counts=per_feature_counts.clone(),
                )
                result, bin_edges = merge_per_feature_percentiles(
                    [shard],
                    n_sae_features=feature_id + 1,
                    occurrence=occurrence.clone(),
                    mag_subsample=mags.clone(),
                    global_mag_min=float(mags.min().item()),
                    percentiles=(50, 90, 99),
                )
                return result[feature_id].percentiles, bin_edges

        fp16_pcts, fp16_edges = _percentiles(force_fp32=False)
        fp32_pcts, fp32_edges = _percentiles(force_fp32=True)

        assert fp16_pcts == fp32_pcts, (
            f"percentiles differ between fp16 and fp32 spill: "
            f"{fp16_pcts} vs {fp32_pcts}"
        )
        assert t.equal(
            t.as_tensor(fp16_edges), t.as_tensor(fp32_edges)
        ), "bin edges differ between fp16 and fp32 spill"

    def test_saved_feature_shards_spill_fp16_and_merge_like_eager(self, tmp_path):
        """Functional path: save_features -> load_features -> optimized_pass1.

        The original f32 mags include values that are not exactly fp16, but the
        shard writer stores them as fp16 and load_features upcasts them. Pass 1
        must therefore take the fp16 spill path, and Pass 2 must match the
        eager percentile implementation on the post-load shard values.
        """
        paths = _write_realistic_saved_shards(tmp_path)
        loaded = [load_features(path) for path in paths]
        feat_ids = t.cat([shard["feature_ids"] for shard in loaded])
        feat_mags = t.cat([shard["feature_mags"] for shard in loaded])
        expected, expected_edges = compute_per_feature_percentiles(
            feat_ids,
            feat_mags,
            percentiles=(50, 90, 99),
        )

        p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
            device="cpu",
        )

        for si in sorted_shards:
            if si.n_activations == 0:
                continue
            assert si.mags_path is not None
            assert _load(si.mags_path).dtype == t.float16

        actual, actual_edges = merge_per_feature_percentiles(
            sorted_shards,
            n_sae_features=p1.n_sae_features,
            occurrence=p1.acc.occurrence,
            mag_subsample=mag_sub,
            global_mag_min=mag_min,
            percentiles=(50, 90, 99),
        )

        assert actual == expected
        assert actual_edges == expected_edges
