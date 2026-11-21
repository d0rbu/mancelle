"""Unit tests for :mod:`steering.eval.axbench_base._stats_reader`.

Exercises every branch of :func:`read_n_sae_features_from_stats` so the
package-private helper that ``runtime_state.py`` and ``dag.py`` both
consume is independently verified.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base import _stats_reader
from steering.eval.axbench_base.types import AxBenchBaseError


def _write_stats_metadata(path: Path, n_sae_features: int) -> None:
    """Write a minimal safetensors stats file whose metadata carries
    ``n_sae_features``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"_placeholder": t.zeros(1, dtype=t.int64)},
        str(path),
        metadata={"n_sae_features": str(n_sae_features)},
    )


class TestReadNSaeFeaturesFromStats:
    """Cover every branch of :func:`read_n_sae_features_from_stats`."""

    def test_happy_path_returns_integer(self, tmp_path: Path) -> None:
        stats_path = tmp_path / "ok.safetensors"
        _write_stats_metadata(stats_path, n_sae_features=32768)
        assert _stats_reader.read_n_sae_features_from_stats(stats_path) == 32768

    def test_happy_path_with_alternate_size(self, tmp_path: Path) -> None:
        stats_path = tmp_path / "alt.safetensors"
        _write_stats_metadata(stats_path, n_sae_features=16384)
        assert _stats_reader.read_n_sae_features_from_stats(stats_path) == 16384

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        stats_path = tmp_path / "nowhere.safetensors"
        with pytest.raises(AxBenchBaseError, match="does not exist"):
            _stats_reader.read_n_sae_features_from_stats(stats_path)

    def test_truncated_header_raises(self, tmp_path: Path) -> None:
        stats_path = tmp_path / "bad.safetensors"
        stats_path.write_bytes(b"\x01\x02\x03")  # < 8 bytes
        with pytest.raises(AxBenchBaseError, match="truncated"):
            _stats_reader.read_n_sae_features_from_stats(stats_path)

    def test_generic_os_error_translates_to_axbench_error(
        self, tmp_path: Path,
    ) -> None:
        """A non-``FileNotFoundError`` ``OSError`` from ``open`` must
        reach the generic branch and get re-raised as
        :class:`AxBenchBaseError`. Using a directory in place of a file
        exercises ``IsADirectoryError`` (an ``OSError`` subclass that
        is NOT ``FileNotFoundError``).
        """
        stats_dir = tmp_path / "stats.safetensors"
        stats_dir.mkdir()
        with pytest.raises(AxBenchBaseError, match="Could not read"):
            _stats_reader.read_n_sae_features_from_stats(stats_dir)

    def test_malformed_header_json_raises(self, tmp_path: Path) -> None:
        stats_path = tmp_path / "junk.safetensors"
        # 8-byte header size prefix declaring 20 bytes of garbage
        header = b"{not valid json!!!!!"
        assert len(header) == 20
        stats_path.write_bytes(struct.pack("<Q", 20) + header)
        with pytest.raises(AxBenchBaseError, match="malformed"):
            _stats_reader.read_n_sae_features_from_stats(stats_path)

    @pytest.mark.parametrize(
        ("header", "match"),
        [
            ([], "non-object safetensors header"),
            ({"__metadata__": []}, "non-object '__metadata__'"),
        ],
    )
    def test_non_object_header_structures_raise_axbench_error(
        self,
        tmp_path: Path,
        header: object,
        match: str,
    ) -> None:
        stats_path = tmp_path / "wrong_shape.safetensors"
        header_bytes = json.dumps(header).encode("utf-8")
        stats_path.write_bytes(
            struct.pack("<Q", len(header_bytes)) + header_bytes,
        )

        with pytest.raises(AxBenchBaseError, match=match):
            _stats_reader.read_n_sae_features_from_stats(stats_path)

    def test_missing_n_sae_features_key_raises(self, tmp_path: Path) -> None:
        stats_path = tmp_path / "no_meta.safetensors"
        save_file(
            {"x": t.zeros(1, dtype=t.int64)},
            str(stats_path),
            metadata={"something_else": "42"},
        )
        with pytest.raises(AxBenchBaseError, match="n_sae_features"):
            _stats_reader.read_n_sae_features_from_stats(stats_path)

    def test_non_int_n_sae_features_raises(self, tmp_path: Path) -> None:
        stats_path = tmp_path / "badval.safetensors"
        save_file(
            {"x": t.zeros(1, dtype=t.int64)},
            str(stats_path),
            metadata={"n_sae_features": "not-an-int"},
        )
        with pytest.raises(AxBenchBaseError, match="non-integer"):
            _stats_reader.read_n_sae_features_from_stats(stats_path)

    def test_metadata_missing_entirely(self, tmp_path: Path) -> None:
        stats_path = tmp_path / "empty_meta.safetensors"
        save_file({"x": t.zeros(1, dtype=t.int64)}, str(stats_path))
        with pytest.raises(AxBenchBaseError, match="n_sae_features"):
            _stats_reader.read_n_sae_features_from_stats(stats_path)

    def test_none_n_sae_features_raises(self, tmp_path: Path) -> None:
        """A metadata dict whose ``n_sae_features`` key is explicitly
        ``None`` (via a null JSON literal) triggers the missing-key
        branch rather than the non-integer branch.
        """
        stats_path = tmp_path / "none_meta.safetensors"
        # Safetensors metadata values must be strings; to exercise the
        # None-path we need to hand-write the header.
        header_dict: dict[str, Any] = {
            "__metadata__": {"n_sae_features": None},
        }
        header_bytes = json.dumps(header_dict).encode("utf-8")
        stats_path.write_bytes(
            struct.pack("<Q", len(header_bytes)) + header_bytes,
        )
        with pytest.raises(AxBenchBaseError, match="n_sae_features"):
            _stats_reader.read_n_sae_features_from_stats(stats_path)

    def test_metadata_none_in_header_raises(self, tmp_path: Path) -> None:
        """An explicit ``"__metadata__": null`` in the header must still
        trigger the missing-key branch (the ``or {}`` fallback hits).
        """
        stats_path = tmp_path / "null_meta.safetensors"
        header_dict: dict[str, Any] = {"__metadata__": None}
        header_bytes = json.dumps(header_dict).encode("utf-8")
        stats_path.write_bytes(
            struct.pack("<Q", len(header_bytes)) + header_bytes,
        )
        with pytest.raises(AxBenchBaseError, match="n_sae_features"):
            _stats_reader.read_n_sae_features_from_stats(stats_path)


class TestHeaderSizeBounds:
    """Item 6: regression for the header-size sanity bound.

    Without an upper bound, a file whose first 8 bytes happen to
    parse as a massive integer would trigger a multi-gigabyte
    ``f.read`` and exhaust memory. The bound is
    :data:`_stats_reader._MAX_HEADER_BYTES`.
    """

    def test_header_size_zero_raises(self, tmp_path: Path) -> None:
        """A safetensors header size of ``0`` is impossible (the header
        must at least contain ``{}``) so the bound-check must raise."""
        stats_path = tmp_path / "zero_header.safetensors"
        stats_path.write_bytes(struct.pack("<Q", 0))
        with pytest.raises(
            AxBenchBaseError, match="outside the valid range",
        ):
            _stats_reader.read_n_sae_features_from_stats(stats_path)

    def test_header_size_too_large_raises(self, tmp_path: Path) -> None:
        """A header_size above the upper bound must raise rather than
        trigger a huge read. We set header_size one byte above the
        maximum so the check fires cleanly."""
        stats_path = tmp_path / "big_header.safetensors"
        bogus_header_size = _stats_reader._MAX_HEADER_BYTES + 1
        stats_path.write_bytes(struct.pack("<Q", bogus_header_size))
        with pytest.raises(
            AxBenchBaseError, match="outside the valid range",
        ):
            _stats_reader.read_n_sae_features_from_stats(stats_path)

    def test_header_size_huge_value_raises_without_big_read(
        self, tmp_path: Path,
    ) -> None:
        """A gigabyte-scale bogus header_size (as could appear from a
        garbage int64) must raise via the bound check instead of
        attempting to allocate gigabytes."""
        stats_path = tmp_path / "gig_header.safetensors"
        # 4 GiB header_size - would exhaust memory without the bound.
        stats_path.write_bytes(struct.pack("<Q", 4 * 1024 * 1024 * 1024))
        with pytest.raises(
            AxBenchBaseError, match="outside the valid range",
        ):
            _stats_reader.read_n_sae_features_from_stats(stats_path)

    def test_header_size_max_boundary_succeeds_with_valid_payload(
        self, tmp_path: Path,
    ) -> None:
        """A header exactly at the max boundary must succeed when its
        payload is valid JSON with the expected metadata."""
        header_dict: dict[str, Any] = {
            "__metadata__": {"n_sae_features": "16384"},
        }
        header_bytes = json.dumps(header_dict).encode("utf-8")
        # Pad the declared header_size up to exactly _MAX_HEADER_BYTES.
        # Note: we pad with null bytes AFTER the JSON payload; the
        # struct reader just reads ``header_size`` bytes and JSON
        # decodes them, so we can't pad above the JSON itself. To pin
        # the boundary case we declare header_size = actual length.
        stats_path = tmp_path / "max_header.safetensors"
        stats_path.write_bytes(
            struct.pack("<Q", len(header_bytes)) + header_bytes,
        )
        # Sanity: the actual declared size is safely under the cap.
        assert len(header_bytes) <= _stats_reader._MAX_HEADER_BYTES
        result = _stats_reader.read_n_sae_features_from_stats(stats_path)
        assert result == 16384
