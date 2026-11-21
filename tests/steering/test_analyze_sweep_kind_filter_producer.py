"""analyze_sweep kind filter semantics.

``kind`` is an optional narrowing filter: ``kind=None`` includes every record
(kinded or not), while an explicit kind returns only records that declare that
kind. Kind-less records (e.g. the training hyperparameter sweep written by
``run_encdec_sweep_fast``) flow through the default ``kind: null`` path.
"""

from __future__ import annotations

import pytest

from scripts.eval.analyze_sweep import filter_by_kind, result_kind

# ---------------------------------------------------------------------------
# Helper: build synthetic records
# ---------------------------------------------------------------------------


def _lockstep_record() -> dict:
    return {
        "kind": "target_real_lockstep",
        "cache_dir": "/fake",
        "seeds": {"1234": {"score": 0.6}},
        "min_val_bpb": 2.1,
        "val_bpb": 2.2,
        "config_label": "lockstep_cfg",
    }


def _pooled_record() -> dict:
    return {
        "kind": "target_real_pooled",
        "cache_dir": "/fake",
        "pooling_op": "mean",
        "seeds": {},
        "min_val_bpb": 2.0,
        "val_bpb": 2.1,
        "config_label": "pooled_cfg",
    }


def _legacy_record() -> dict:
    """JSONL record without kind field — legacy sweep shape."""
    return {
        "seeds": {"1234": {"score": 0.5}},
        "min_val_bpb": 2.3,
        "val_bpb": 2.4,
        "config_label": "legacy_cfg",
    }


# ---------------------------------------------------------------------------
# Test 1: filter_by_kind returns correct records when kind field IS present
# ---------------------------------------------------------------------------


def test_analyze_sweep_kind_filter_returns_lockstep_entries():
    """filter_by_kind with kind=target_real_lockstep returns only lockstep records."""
    records = [_lockstep_record(), _pooled_record()]
    filtered = filter_by_kind(records, "target_real_lockstep")
    assert len(filtered) == 1, (
        f"Expected 1 lockstep record; got {len(filtered)}: "
        f"{[r.get('config_label') for r in filtered]}"
    )
    assert filtered[0]["kind"] == "target_real_lockstep"


def test_analyze_sweep_kind_filter_returns_pooled_entries():
    """filter_by_kind with kind=target_real_pooled returns only pooled records."""
    records = [_lockstep_record(), _pooled_record()]
    filtered = filter_by_kind(records, "target_real_pooled")
    assert len(filtered) == 1
    assert filtered[0]["kind"] == "target_real_pooled"


def test_analyze_sweep_kind_filter_all_explicit_records_when_kind_is_none():
    """filter_by_kind with kind=None returns every explicitly-kinded variant."""
    records = [_lockstep_record(), _pooled_record()]
    filtered = filter_by_kind(records, None)
    assert filtered == records


# ---------------------------------------------------------------------------
# Test 2: filter_by_kind returns empty when no records match
# ---------------------------------------------------------------------------


def test_analyze_sweep_kind_filter_returns_empty_when_no_match():
    """When no records carry the requested kind, result is empty (not None)."""
    records = [_pooled_record(), _pooled_record()]
    filtered = filter_by_kind(records, "target_real_lockstep")
    assert filtered == [], (
        f"Expected empty list when no records match; got {filtered}"
    )


def test_analyze_sweep_kind_filter_records_without_kind_raise():
    """Records without kind field are rejected instead of treated as plain."""
    record = _legacy_record()
    with pytest.raises(ValueError, match="missing required 'kind'"):
        result_kind(record)


def test_filter_by_kind_explicit_kind_skips_kindless_records():
    """An explicit kind filter drops kind-less records without raising.

    ``kind`` only narrows records that declare a kind; a kind-less record
    simply does not match the requested kind (it is not an error). A matching
    kinded record alongside it is still returned.
    """
    pooled = _pooled_record()
    records = [pooled, _legacy_record()]
    filtered = filter_by_kind(records, "target_real_pooled")
    assert filtered == [pooled]


def test_filter_by_kind_none_includes_kindless_records():
    """kind=None is a true no-op: every record passes through, kinded or not.

    Regression: analyze_sweep used to exit(1) by default on its own producer's
    output. run_encdec_sweep_fast writes kind-less records and the default
    config sets ``kind: null``, so the documented invocation must not reject
    them.
    """
    records = [_lockstep_record(), _legacy_record()]
    assert filter_by_kind(records, None) == records


def _sweep_producer_record() -> dict:
    """Mirror the kind-less JSONL row written by ``_serialize_result``.

    Keys match ``scripts/train/run_encdec_sweep_fast.py::_serialize_result``;
    notably there is no ``kind`` field — these are training hyperparameter
    sweep rows, not eval-conditioning variants.
    """
    return {
        "model_tag": "d12",
        "config_label": "geo/ns_g/fd_b/fpw_0.5",
        "val_bpb": 2.4,
        "min_val_bpb": 2.3,
        "total_training_time": 12.0,
        "peak_memory_mib": 1024.0,
        "startup_time": 3.0,
        "error": None,
        "fs_distribution": "geometric",
        "fs_mean": 4.0,
        "fs_std": 1.0,
        "ns_mean": 2.0,
        "fd_mode": "balanced",
        "fd_neg_delta_ratio": 0.5,
        "fp_weight": 0.5,
        "config_index": 0,
        "timestamp": "2026-05-25T00:00:00+00:00",
    }


def test_filter_by_kind_none_passes_through_sweep_producer_records():
    """The default ``kind: null`` run must accept run_encdec_sweep_fast output."""
    records = [_sweep_producer_record(), _sweep_producer_record()]
    assert filter_by_kind(records, None) == records
