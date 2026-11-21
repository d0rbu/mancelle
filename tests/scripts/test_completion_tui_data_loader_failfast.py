"""Issue #521: completion TUI preserves None-return via membership pre-check.

After Task 1, ``FeatureIndex.lookup`` RAISES ``KeyError`` on an absent doc
instead of returning ``None``. The completion TUI's ``raw_lookup`` /
``total_segments`` / ``load_document`` document a ``None``-return for an
out-of-range doc (a legitimate interactive-tool contract). These tests pin
that the TUI pre-checks ``self._doc_ids`` membership so that ``lookup`` only
fires for present docs — preserving the ``None``-return without swallowing a
genuine-corruption ``KeyError``.

The loader is constructed via ``__new__`` + hand-set attributes (mirroring
tests/steering/test_sft_features_failfast.py) so the unit test stays light and
does not require real parquet/feature-shard fixtures.
"""

from __future__ import annotations

from typing import cast

from scripts.completion_tui.data_loader import DocumentLoader
from steering.conditional.feature_index import FeatureIndex


class _StubIndex:
    """Stand-in for FeatureIndex that mirrors the post-Task-1 lookup contract."""

    def __init__(self, present):
        self._present = present  # dict[int, list]
        self.segment_size = 128

    def lookup(self, shard_stem, doc_idx):
        if doc_idx not in self._present:
            raise KeyError(
                f"Document {doc_idx} not found in shard {shard_stem!r} "
                f"(shard has {len(self._present)} docs)"
            )
        return self._present[doc_idx]

    def doc_ids_for_shard(self, shard_stem):
        return sorted(self._present)


def _loader(present_docs):
    loader = DocumentLoader.__new__(DocumentLoader)
    loader._feature_index = cast(FeatureIndex, _StubIndex(present_docs))
    loader._val_stem = "shard_00000"
    loader._doc_ids = sorted(present_docs)
    loader._texts = ["text"] * (max(present_docs) + 5 if present_docs else 5)
    return loader


def test_raw_lookup_absent_returns_none_not_keyerror():
    loader = _loader({0: [object()], 1: [object()]})
    assert loader.raw_lookup(2) is None  # absent -> None (pre-check), no KeyError


def test_raw_lookup_present_returns_segments():
    segs = [object()]
    loader = _loader({0: segs})
    assert loader.raw_lookup(0) is segs


def test_total_segments_absent_returns_none():
    loader = _loader({0: [object(), object()]})
    assert loader.total_segments(5) is None


def test_total_segments_present_returns_count():
    loader = _loader({0: [object(), object(), object()]})
    assert loader.total_segments(0) == 3
