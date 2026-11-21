"""Property-based invariants for statistics and percentile tables.

- ``magnitude_to_percentile`` rank monotonicity.
- ``compute_percentile_checksum`` determinism (insertion-order
  invariance + repeated-call invariance).
- ``discover_overlay`` prefix-match unidirectionality.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
    magnitude_to_percentile,
    percentile_to_uint8,
    uint8_to_percentile,
)
from steering.overlay_io import (
    MANIFEST_FILENAME,
    OverlayKind,
    discover_overlay,
    overlay_dir,
    write_manifest,
)
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_per_feature_stats_with_monotone_pcts(
    feature_ids: list[int],
    *,
    seed: int = 0,
) -> dict[str, PerFeatureStat]:
    """Build a small per_feature_stats dict with monotone p1..p99 rows."""
    g = t.Generator().manual_seed(seed)
    out: dict[str, PerFeatureStat] = {}
    for fid in feature_ids:
        # Sample 99 monotone increasing percentile values in (0, 100).
        base = t.rand(99, generator=g) * 100.0
        sorted_vals = base.sort().values
        pcts = {f"p{k}": float(sorted_vals[k - 1].item()) for k in range(1, 100)}
        out[str(fid)] = PerFeatureStat(
            count=10,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.1,
            min_mag=float(sorted_vals[0].item()),
            max_mag=float(sorted_vals[-1].item()),
            median_mag=float(sorted_vals[49].item()),
            percentiles=pcts,
        )
    return out


# ---------------------------------------------------------------------------
# T10 — magnitude_to_percentile rank monotonicity
# ---------------------------------------------------------------------------


class TestMagnitudeToPercentileRankMonotone:
    """``m → p → uint8 → p → m`` is rank-preserving (modulo uint8 quantization)."""

    @given(
        mag_a=st.floats(min_value=-10.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        mag_b=st.floats(min_value=-10.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, derandomize=True)
    def test_codec_chain_preserves_rank(self, mag_a: float, mag_b: float) -> None:
        """For any two magnitudes a, b on the SAME feature, mag_a ≤ mag_b ⇒ codec(a) ≤ codec(b)."""
        per_feature_stats = _make_per_feature_stats_with_monotone_pcts([0])
        table = build_percentile_table(per_feature_stats, n_sae_features=1)

        feature_ids = t.tensor([0, 0], dtype=t.int64)
        feature_mags = t.tensor([mag_a, mag_b], dtype=t.float32)
        pcts = magnitude_to_percentile(feature_ids, feature_mags, table)
        encoded = percentile_to_uint8(pcts)
        decoded = uint8_to_percentile(encoded)

        if mag_a <= mag_b:
            # uint8 quantization can collapse adjacent magnitudes — contract is ≤, not <.
            assert decoded[0].item() <= decoded[1].item(), (
                f"codec must be monotone: mag_a={mag_a}, mag_b={mag_b}, "
                f"pcts={pcts.tolist()}, decoded={decoded.tolist()}"
            )
        else:
            assert decoded[0].item() >= decoded[1].item()


# ---------------------------------------------------------------------------
# T11 — compute_percentile_checksum determinism
# ---------------------------------------------------------------------------


class TestPercentileChecksumDeterminism:
    """Checksum is identical across repeated calls and insertion-order permutations."""

    def test_checksum_repeated_calls_match(self) -> None:
        per_feature_stats = _make_per_feature_stats_with_monotone_pcts([0, 1, 2])
        table = build_percentile_table(per_feature_stats, n_sae_features=4)
        c1 = compute_percentile_checksum(table)
        c2 = compute_percentile_checksum(table)
        c3 = compute_percentile_checksum(table)
        assert c1 == c2 == c3, (
            f"checksum must be deterministic across repeated calls; got {c1}, {c2}, {c3}"
        )

    def test_checksum_invariant_to_insertion_order(self) -> None:
        """``build_percentile_table`` is indexed by ``int(fid)``, so dict insertion order should not affect the checksum."""
        # Build the same logical table from two differently-ordered dicts.
        stats_a = _make_per_feature_stats_with_monotone_pcts([0, 1, 2], seed=0)
        # Re-construct an order-permuted dict with the SAME logical content.
        keys = list(stats_a.keys())
        permuted = {k: stats_a[k] for k in reversed(keys)}

        table_a = build_percentile_table(stats_a, n_sae_features=4)
        table_b = build_percentile_table(permuted, n_sae_features=4)
        c_a = compute_percentile_checksum(table_a)
        c_b = compute_percentile_checksum(table_b)
        # Tables are indexed by feature id, so they must be byte-identical.
        assert t.equal(table_a, table_b), (
            "tables must be byte-identical regardless of insertion order"
        )
        assert c_a == c_b, (
            f"insertion-order permuted dicts must produce the same checksum; "
            f"got {c_a} vs {c_b}"
        )

    def test_checksum_changes_when_data_changes(self) -> None:
        """Sanity: a non-trivial change MUST change the checksum."""
        stats_a = _make_per_feature_stats_with_monotone_pcts([0, 1], seed=0)
        stats_b = _make_per_feature_stats_with_monotone_pcts([0, 1], seed=1)
        table_a = build_percentile_table(stats_a, n_sae_features=2)
        table_b = build_percentile_table(stats_b, n_sae_features=2)
        c_a = compute_percentile_checksum(table_a)
        c_b = compute_percentile_checksum(table_b)
        assert c_a != c_b, (
            "different percentile rows MUST yield different checksums"
        )


# ---------------------------------------------------------------------------
# T12 — discover_overlay prefix-match unidirectionality
# ---------------------------------------------------------------------------


class TestDiscoverOverlayPrefixMatch:
    """The caller's checksum must be a prefix of the manifest's full checksum.

    ``manifest_checksum.startswith(caller_checksum)`` is the contract; reversing
    the direction (manifest prefix matching caller) is unsafe because it would
    validate against truncated data.
    """

    def test_caller_full_hash_matches_manifest_full_hash(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        full_hash = "a" * 64
        d = overlay_dir(feature_dir, OverlayKind.PERCENTILE, full_hash)
        d.mkdir(parents=True, exist_ok=True)
        write_manifest(d, full_hash, OverlayKind.PERCENTILE)
        result = discover_overlay(feature_dir, OverlayKind.PERCENTILE, full_hash)
        assert result == d

    def test_caller_prefix_matches_manifest_full_hash(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        full_hash = "b" * 64
        d = overlay_dir(feature_dir, OverlayKind.PERCENTILE, full_hash)
        d.mkdir(parents=True, exist_ok=True)
        write_manifest(d, full_hash, OverlayKind.PERCENTILE)
        # Caller passes a 16-char prefix.
        result = discover_overlay(feature_dir, OverlayKind.PERCENTILE, full_hash[:16])
        assert result == d

    def test_caller_full_hash_rejects_manifest_prefix_only(self, tmp_path: Path) -> None:
        """Reverse direction is rejected: a manifest with a TRUNCATED checksum cannot match a fuller caller checksum."""
        # Build the directory using the prefix-style checksum so the dir lives
        # at the expected ``pct/<prefix>`` location.
        feature_dir = tmp_path / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        full_hash = "c" * 64
        prefix = full_hash[:16]
        # Construct a manifest at the prefix location whose CHECKSUM field is
        # only the prefix — i.e. a truncated manifest.
        d = overlay_dir(feature_dir, OverlayKind.PERCENTILE, full_hash)
        d.mkdir(parents=True, exist_ok=True)
        manifest_path = d / MANIFEST_FILENAME
        manifest_path.write_text(json.dumps({
            "checksum": prefix,  # truncated! NOT the full 64-char hash.
            "kind": OverlayKind.PERCENTILE.value,
            "checksum_prefix": prefix,
        }))
        # Caller passes the FULL hash. ``manifest_checksum.startswith(full_hash)``
        # is False, so discover_overlay must return None.
        result = discover_overlay(feature_dir, OverlayKind.PERCENTILE, full_hash)
        assert result is None, (
            "discover_overlay must reject when manifest checksum is shorter than caller's"
        )

    def test_empty_caller_checksum_returns_none(self, tmp_path: Path) -> None:
        """Empty checksum is rejected up front."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        result = discover_overlay(feature_dir, OverlayKind.PERCENTILE, "")
        assert result is None
