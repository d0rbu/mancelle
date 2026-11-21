"""Slice 04 — ``_save_percentile_dict`` save/load round-trip precision regression.

Pins the documented load-time rounding to 4 decimal digits (manifest §2.4 /
map P3 #14). The save side writes float64 as-is; the load side rounds to 4
digits. ``compute_statistics`` typically pre-rounds via ``_percentile_dict
(round_digits=…)`` so the asymmetry is invisible at the orchestrator level —
but raw use of the helpers DOES surface it. These tests pin the current
behaviour so a future precision tightening (or loosening) is detected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

from steering.stats_io import _load_percentile_dict, _save_percentile_dict


class TestSavePercentileDictRoundTrip:
    """``_save_percentile_dict`` writes float64; ``_load_percentile_dict`` rounds to 4 digits."""

    def test_int_values_preserved_exactly(self, tmp_path: Path) -> None:
        """Integer count percentiles round-trip without loss (always within 4-digit precision)."""
        tensors: dict[str, t.Tensor] = {}
        metadata: dict[str, str] = {}
        pct = {f"p{p}": float(p) for p in (1, 10, 50, 90, 99)}
        _save_percentile_dict(tensors, metadata, "count_pct", pct)

        # Simulate save→load via in-memory tensor dict.
        loaded = _load_percentile_dict(tensors, metadata, "count_pct")
        for k, v in pct.items():
            assert loaded[k] == pytest.approx(v, abs=1e-9), (
                f"int-valued percentile {k} must round-trip exactly; got {loaded[k]} for {v}"
            )

    def test_float_values_within_4_digit_precision_preserved(self, tmp_path: Path) -> None:
        """Values with ≤ 4 decimal digits of precision survive round-trip exactly."""
        tensors: dict[str, t.Tensor] = {}
        metadata: dict[str, str] = {}
        pct = {"p1": 0.0001, "p50": 1.5000, "p99": 99.9999}
        _save_percentile_dict(tensors, metadata, "mag_pct", pct)
        loaded = _load_percentile_dict(tensors, metadata, "mag_pct")
        for k, v in pct.items():
            assert loaded[k] == pytest.approx(v, abs=1e-9), (
                f"4-digit-precision percentile {k} must round-trip; got {loaded[k]}"
            )

    def test_subdigit_precision_is_rounded_on_load(self, tmp_path: Path) -> None:
        """Pin: load-time rounds to 4 digits — values past 4 decimal digits are coarsened."""
        tensors: dict[str, t.Tensor] = {}
        metadata: dict[str, str] = {}
        # Value with 5 decimal digits of precision.
        pct = {"p50": 1.234567}
        _save_percentile_dict(tensors, metadata, "mag_pct", pct)
        loaded = _load_percentile_dict(tensors, metadata, "mag_pct")
        # Pinned current behaviour: round to 4 digits ⇒ 1.2346.
        assert loaded["p50"] == pytest.approx(1.2346, abs=1e-9), (
            f"load must round to 4 digits; got {loaded['p50']!r} (expected 1.2346)"
        )

    def test_empty_dict_is_no_op(self, tmp_path: Path) -> None:
        """Empty pct dict is not persisted; load returns empty dict."""
        tensors: dict[str, t.Tensor] = {}
        metadata: dict[str, str] = {}
        _save_percentile_dict(tensors, metadata, "mag_pct", {})
        # No keys recorded.
        assert "mag_pct_keys" not in metadata
        assert "mag_pct_values" not in tensors
        assert _load_percentile_dict(tensors, metadata, "mag_pct") == {}

    def test_none_dict_is_no_op(self, tmp_path: Path) -> None:
        """``None`` pct dict is not persisted; load returns empty dict."""
        tensors: dict[str, t.Tensor] = {}
        metadata: dict[str, str] = {}
        _save_percentile_dict(tensors, metadata, "mag_pct", None)
        assert "mag_pct_keys" not in metadata
        assert "mag_pct_values" not in tensors
        assert _load_percentile_dict(tensors, metadata, "mag_pct") == {}
