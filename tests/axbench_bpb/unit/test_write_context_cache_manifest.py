# tests/axbench_bpb/unit/test_write_context_cache_manifest.py
"""Unit tests for write_context_cache_manifest helper.

Commit 24680f51 split the manifest write out of build_context_cache
so rank 0 can call it AFTER a barrier under DDP. The helper:

1. Reads back every per-prefix safetensors file
2. Builds {prefix_hash: {n_segments, n_features_total}}
3. Writes manifest.json via tmp + rename (atomic on POSIX)

These tests pin all three invariants at the unit layer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from steering.eval.axbench_bpb.context_cache import (
    load_manifest,
    serialize_context_segments,
    write_context_cache_manifest,
)
from steering.eval.axbench_bpb.paths import OutputPaths
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.target_conditioning import ContextFeatureSegment


def _prefix_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _make_paths(tmp_path: Path) -> OutputPaths:
    return OutputPaths(output_root=tmp_path, run_tag="test")


def _write_segs(
    out_paths: OutputPaths, prefix_hash: str,
    feature_ids: tuple[int, ...], buckets: tuple[int, ...],
) -> None:
    """Helper: serialize a single one-segment fixture for a given hash."""
    segs = (
        ContextFeatureSegment(
            feature_ids=feature_ids, buckets=buckets,
            char_start=0, char_end=10,
        ),
    )
    serialize_context_segments(segs, out_paths.context_cache_file(prefix_hash))


def test_manifest_content_matches_serialized_segments(tmp_path):
    """Manifest entries reflect the actual segment count and feature
    count of the underlying safetensors files."""
    paths = _make_paths(tmp_path)
    unique = {
        "alpha": _prefix_hash("alpha"),
        "beta": _prefix_hash("beta"),
        "gamma": _prefix_hash("gamma"),
    }
    _write_segs(paths, unique["alpha"], (1, 2, 3), (0, 1, 0))
    _write_segs(paths, unique["beta"], (4,), (1,))
    _write_segs(paths, unique["gamma"], (5, 6), (1, 0))

    write_context_cache_manifest(unique_prefixes=unique, output_paths=paths)

    manifest = load_manifest(paths)
    assert set(manifest.keys()) == set(unique.values())
    assert manifest[unique["alpha"]] == {
        "n_segments": 1, "n_features_total": 3,
    }
    assert manifest[unique["beta"]] == {
        "n_segments": 1, "n_features_total": 1,
    }
    assert manifest[unique["gamma"]] == {
        "n_segments": 1, "n_features_total": 2,
    }


def test_manifest_write_is_atomic(tmp_path):
    """tmp + rename pattern: after success, final file exists, no .tmp
    sibling, no other partial files in the manifest directory."""
    paths = _make_paths(tmp_path)
    unique = {"x": _prefix_hash("x")}
    _write_segs(paths, unique["x"], (1,), (0,))

    write_context_cache_manifest(unique_prefixes=unique, output_paths=paths)

    manifest_path = paths.context_cache_manifest()
    assert manifest_path.exists()
    siblings = list(manifest_path.parent.glob("manifest.json*"))
    # Only the final file should be present; no .tmp leftovers.
    assert siblings == [manifest_path], (
        f"Atomic-write contract violated: found extra siblings {siblings}"
    )


def test_manifest_write_returns_path(tmp_path):
    """Return value is the manifest path so the caller (the legacy
    single-process branch of build_context_cache) can chain it."""
    paths = _make_paths(tmp_path)
    unique = {"x": _prefix_hash("x")}
    _write_segs(paths, unique["x"], (1,), (0,))

    result = write_context_cache_manifest(
        unique_prefixes=unique, output_paths=paths,
    )
    assert result == paths.context_cache_manifest()


def test_manifest_write_iterates_deterministically(tmp_path):
    """The helper sorts by prefix_hash for a stable iteration order so
    JSON output is byte-identical regardless of the input dict's
    insertion order.

    To actually exercise the ``sorted(...)`` call (and not just dict
    insertion-order stability), we feed the SAME contents in two
    DIFFERENT insertion orders. A regression that drops ``sorted(...)``
    and uses raw dict iteration would produce two distinct JSON files."""
    paths = _make_paths(tmp_path)
    items = [(f"k{i}", _prefix_hash(f"k{i}")) for i in range(5)]
    for _, v in items:
        _write_segs(paths, v, (1,), (0,))

    # Insertion order 1: as constructed.
    unique_a = dict(items)
    write_context_cache_manifest(unique_prefixes=unique_a, output_paths=paths)
    raw_a = paths.context_cache_manifest().read_text()

    # Insertion order 2: reversed.
    unique_b = dict(reversed(items))
    write_context_cache_manifest(unique_prefixes=unique_b, output_paths=paths)
    raw_b = paths.context_cache_manifest().read_text()

    # Sanity: insertion orders ARE different.
    assert list(unique_a.keys()) != list(unique_b.keys()), (
        "Test setup error: insertion orders should differ"
    )
    # Despite different insertion orders, manifest JSON is byte-identical.
    assert raw_a == raw_b, (
        "Manifest output depends on input dict insertion order — "
        "sorted(...) call in write_context_cache_manifest is broken or missing"
    )


def test_manifest_write_raises_when_safetensors_missing(tmp_path):
    """If the manifest references a prefix_hash whose safetensors file
    is missing, load_context_segments raises an underlying error (we
    don't try to swallow it — missing files indicate a logic bug in
    the caller's barrier ordering, not a recoverable state)."""
    paths = _make_paths(tmp_path)
    unique = {"missing": _prefix_hash("missing")}
    # NO _write_segs call — the file is absent.

    # safetensors.torch.load_file raises FileNotFoundError when the
    # underlying file is missing (verified via load_context_segments,
    # which does not wrap the exception). Pin the concrete class so a
    # future regression that raises an unrelated error (KeyError, etc.)
    # does not silently pass this test.
    with pytest.raises(FileNotFoundError):
        write_context_cache_manifest(
            unique_prefixes=unique, output_paths=paths,
        )


def test_load_manifest_round_trips_what_writer_produces(tmp_path):
    """load_manifest reads back exactly what write_context_cache_manifest
    wrote. Pins the read/write pair as a contract."""
    paths = _make_paths(tmp_path)
    unique = {
        "first": _prefix_hash("first"),
        "second": _prefix_hash("second"),
    }
    _write_segs(paths, unique["first"], (10, 20), (1, 0))
    _write_segs(paths, unique["second"], (30,), (0,))

    write_context_cache_manifest(unique_prefixes=unique, output_paths=paths)

    raw_text = paths.context_cache_manifest().read_text()
    manifest = json.loads(raw_text)
    via_loader = load_manifest(paths)

    assert manifest == via_loader


def test_load_manifest_raises_when_missing(tmp_path):
    """load_manifest raises AxBenchBpbError when the manifest is absent
    so downstream stages fail fast with a clear message."""
    paths = _make_paths(tmp_path)
    with pytest.raises(AxBenchBpbError, match="manifest"):
        load_manifest(paths)
