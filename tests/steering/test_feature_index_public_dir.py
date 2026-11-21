"""Slice-10 audit: ``FeatureIndex.feature_dir`` public read-only property.

Pre-fix: ``encoder_decoder/dataloader.py`` reached into
``feature_index._feature_dir`` (private) for the cross-stage segment_size
verification check. Private-attr reads couple slices and break if slice 7
ever renames the underlying field.

Fix: a public ``feature_dir`` property was added to :class:`FeatureIndex`
(strictly additive, read-only). The dataloader call site now uses it.

Covers plan T-11.

See manifest §7 / slice-10 map P2-A.
"""

from __future__ import annotations

import inspect

from steering.conditional.feature_index import FeatureIndex


def test_public_feature_dir_property_exists():
    """``FeatureIndex.feature_dir`` is a property (not a method or attribute)."""
    descr = inspect.getattr_static(FeatureIndex, "feature_dir")
    assert isinstance(descr, property), (
        f"FeatureIndex.feature_dir must be a property, got {type(descr).__name__}"
    )


def test_public_feature_dir_returns_private_field(tmp_path):
    """The public property returns the same value as the private attribute.

    We use a fresh empty directory: ``FeatureIndex._scan`` will record it as
    ``_feature_dir`` even though it contains zero shards. (The constructor
    is happy with an empty dir; ``n_shards`` will simply be zero.)
    """
    # Create an empty directory so the constructor's _scan call succeeds.
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()

    fi = FeatureIndex(feature_dir=str(feat_dir))

    # Public reads the same path the private attribute holds.
    assert fi.feature_dir == fi._feature_dir
    # And n_shards is zero, but that doesn't matter for this test — we only
    # care about the property surface.
    assert fi.n_shards == 0


def test_public_feature_dir_returned_path_is_pathlike(tmp_path):
    """Property returns a ``Path | None`` per its annotation."""
    from pathlib import Path

    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    fi = FeatureIndex(feature_dir=str(feat_dir))

    fd = fi.feature_dir
    # Either a Path or None; never a str.
    assert fd is None or isinstance(fd, Path)


def test_public_feature_dir_is_read_only(tmp_path):
    """Setting the property raises (read-only contract)."""
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    fi = FeatureIndex(feature_dir=str(feat_dir))

    try:
        fi.feature_dir = tmp_path  # type: ignore[misc]  # ty: ignore[invalid-assignment]
    except AttributeError:
        return  # expected
    raise AssertionError("FeatureIndex.feature_dir must be read-only")


def test_dataloader_uses_public_property():
    """Static-import check: ``encoder_decoder/dataloader.py`` no longer
    references the private ``_feature_dir``.

    A grep test catches regressions where someone re-introduces the private
    access pattern. Slice-10 P2-A converged on ``feature_dir`` (public).
    """
    from steering.conditional.encoder_decoder import dataloader as dl

    src = inspect.getsource(dl)
    # The cross-stage check must read the public property.
    assert "feature_index.feature_dir" in src, (
        "dataloader.py must read feature_index.feature_dir (public)"
    )
    # And must NOT reach into the private attribute.
    assert "feature_index._feature_dir" not in src, (
        "dataloader.py must not reach into the private _feature_dir"
    )
