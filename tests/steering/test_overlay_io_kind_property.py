"""``OverlayKind`` enum and manifest-kind round trip.

Pin the ``OverlayKind.PACKED_ROW`` value and the round-trip property of
``write_manifest`` → ``read_manifest`` for every supported ``OverlayKind``
value.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.overlay_io import (
    MANIFEST_FILENAME,
    OverlayKind,
    read_manifest,
    write_manifest,
)

FAKE_CHECKSUM = "abcdef0123456789" + "0" * 48  # 64-char hex


class TestPackedRowEnumValueExists:
    def test_packed_row_value_string(self):
        # A1: introduces a label-only enum value used by precompute_packed_rows.
        assert OverlayKind.PACKED_ROW.value == "packed_rows"

    def test_distinct_from_other_kinds(self):
        # The values must not collide across supported overlay kinds.
        all_values = {kind.value for kind in OverlayKind}
        assert all_values == {
            "pct",
            "seg_boundaries",
            "packed_rows",
            "grad",
        }, (
            f"unexpected enum values: {all_values!r}"
        )


class TestOverlayDirRejectsPackedRow:
    """``overlay_dir`` builds canonical paths for PERCENTILE / SEG_BOUNDARY
    only — packed-row paths are constructed manually by the precompute script
    (4-level nested layout). The function must raise a clear error if a caller
    accidentally asks for a packed-row path via ``overlay_dir``.
    """

    def test_overlay_dir_packed_row_raises(self, tmp_path):
        from steering.overlay_io import overlay_dir

        with pytest.raises(ValueError, match="PACKED_ROW"):
            overlay_dir(tmp_path / "features", OverlayKind.PACKED_ROW, FAKE_CHECKSUM)

class TestManifestKindRoundTrip:
    """``write_manifest`` writes the enum value as the JSON ``kind`` field;
    ``read_manifest`` returns it untouched. Property: every supported kind
    round-trips losslessly.
    """

    @pytest.mark.parametrize("kind", list(OverlayKind))
    def test_kind_round_trip(self, tmp_path, kind):
        d = tmp_path / "manifest_dir"
        d.mkdir()
        write_manifest(d, FAKE_CHECKSUM, kind)
        manifest = read_manifest(d)
        assert manifest is not None
        assert manifest["kind"] == kind.value
        assert manifest["checksum"] == FAKE_CHECKSUM

    def test_packed_row_kind_serializes_as_packed_rows(self, tmp_path):
        # Regression for slice 05 P1 #1 (consolidation manifest §4.9): the
        # packed-row precompute used to mis-label dirs as "pct".
        d = tmp_path / "manifest_dir"
        d.mkdir()
        write_manifest(d, FAKE_CHECKSUM, OverlayKind.PACKED_ROW)
        raw = json.loads((d / MANIFEST_FILENAME).read_text())
        assert raw["kind"] == "packed_rows"


class TestManifestPropertyHypothesis:
    @settings(
        derandomize=True, max_examples=200,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        kind=st.sampled_from(list(OverlayKind)),
        # Realistic 64-char hex checksum
        checksum=st.text(
            alphabet="0123456789abcdef", min_size=64, max_size=64,
        ),
    )
    def test_manifest_round_trip_property(self, tmp_path, kind, checksum):
        d = tmp_path / "manifest_dir"
        d.mkdir(exist_ok=True)
        write_manifest(d, checksum, kind)
        manifest = read_manifest(d)
        assert manifest is not None
        assert manifest["kind"] == kind.value
        assert manifest["checksum"] == checksum
