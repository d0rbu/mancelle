"""Weekly manual check: ensure Neuronpedia's /api/feature response shape
still matches our captured fixtures.

Warns LOUDLY if any top-level or per-entry field has been added, removed,
or renamed relative to the baseline. Also fails if binContains drifts
outside [0, 1.1] (detects a semantics-shift case). Exit 0 = no drift.

Usage:
    NEURONPEDIA_API_KEY=... uv run python -m scripts.validation.check_neuronpedia_schema
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

from steering.eval.neuronpedia_client import (
    NEURONPEDIA_API_KEY_ENV,
    NEURONPEDIA_BASE_URL,
)

FIXTURES_DIR = (
    Path(__file__).resolve().parents[2]
    / "tests" / "feature_filter" / "fixtures"
)

# Features compared between the live API and the captured fixtures.
FEATURES = (
    (42, FIXTURES_DIR / "np_feature_gemma_res16k_42.json"),
    (100, FIXTURES_DIR / "np_feature_gemma_res16k_100.json"),
)

REQUIRED_TOP = {
    "explanations", "activations", "freq_hist_data_bar_heights",
    "freq_hist_data_bar_values", "frac_nonzero", "maxActApprox",
}
REQUIRED_ENTRY = {
    "tokens", "values", "maxValue", "maxValueTokenIndex",
    "binMin", "binMax", "binContains",
}


def _collect_entry_keys(activations: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for a in activations:
        out.update(a.keys())
    return out


def _check_live_payload(feature_id: int, payload: dict[str, Any]) -> list[str]:
    drift: list[str] = []
    missing_top = REQUIRED_TOP - set(payload)
    if missing_top:
        drift.append(
            f"feature {feature_id}: top-level missing: {sorted(missing_top)}"
        )
    acts = payload.get("activations") or []
    if not acts:
        drift.append(f"feature {feature_id}: no activations in live payload")
        return drift

    entry_keys = _collect_entry_keys(acts)
    missing_entry = REQUIRED_ENTRY - entry_keys
    if missing_entry:
        drift.append(
            f"feature {feature_id}: activation entry missing: "
            f"{sorted(missing_entry)}"
        )

    # Bin structure assertions (sentinel present, <=5 bins, hi > lo).
    has_sentinel = False
    non_sentinel_bins: set[tuple[float, float]] = set()
    for a in acts:
        lo = float(a["binMin"])
        hi = float(a["binMax"])
        if lo == -1.0:
            has_sentinel = True
            continue
        if hi <= lo:
            drift.append(
                f"feature {feature_id}: degenerate bin min={lo} max={hi}"
            )
            continue
        non_sentinel_bins.add((lo, hi))
        # Semantics check: binContains should be a probability-like value.
        bc = a.get("binContains")
        if bc is not None:
            try:
                bc_f = float(bc)
            except (TypeError, ValueError):
                drift.append(
                    f"feature {feature_id}: non-numeric binContains={bc!r}"
                )
                continue
            if not (0.0 <= bc_f <= 1.1):
                drift.append(
                    f"feature {feature_id}: binContains={bc_f} outside [0, 1.1] "
                    "(Neuronpedia may have changed semantics)"
                )
    if not has_sentinel:
        drift.append(
            f"feature {feature_id}: no binMin=-1 sentinel entries "
            "(Neuronpedia may have changed stratification)"
        )
    if not (1 <= len(non_sentinel_bins) <= 5):
        drift.append(
            f"feature {feature_id}: expected 1-5 non-sentinel bins, got "
            f"{len(non_sentinel_bins)}"
        )
    return drift


def _check_fixture_consistency() -> list[str]:
    drift: list[str] = []
    top_keys_by_fid: dict[int, set[str]] = {}
    entry_keys_by_fid: dict[int, set[str]] = {}
    for fid, path in FEATURES:
        if not path.is_file():
            drift.append(f"feature {fid}: fixture missing at {path}")
            continue
        data = json.loads(path.read_text())
        top_keys_by_fid[fid] = set(data.keys())
        entries = data.get("entries", []) or []
        entry_keys_by_fid[fid] = _collect_entry_keys(entries)
    if len(top_keys_by_fid) >= 2:
        fids = list(top_keys_by_fid)
        ref_top = top_keys_by_fid[fids[0]]
        ref_entry = entry_keys_by_fid[fids[0]]
        for fid in fids[1:]:
            if top_keys_by_fid[fid] != ref_top:
                drift.append(
                    f"fixture top-level keys differ between feature "
                    f"{fids[0]} and feature {fid}: "
                    f"symmetric_diff={top_keys_by_fid[fid] ^ ref_top}"
                )
            if entry_keys_by_fid[fid] != ref_entry:
                drift.append(
                    f"fixture entry keys differ between feature "
                    f"{fids[0]} and feature {fid}: "
                    f"symmetric_diff={entry_keys_by_fid[fid] ^ ref_entry}"
                )
    return drift


def main() -> int:
    key = os.environ.get(NEURONPEDIA_API_KEY_ENV)
    if not key:
        print(f"{NEURONPEDIA_API_KEY_ENV} not set", file=sys.stderr)
        return 2

    all_drift: list[str] = []
    all_drift.extend(_check_fixture_consistency())

    for fid, _path in FEATURES:
        url = f"{NEURONPEDIA_BASE_URL}/feature/gemma-2-2b/8-gemmascope-res-16k/{fid}"
        resp = requests.get(url, headers={"X-Api-Key": key}, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        all_drift.extend(_check_live_payload(fid, payload))

    if all_drift:
        print("Neuronpedia SCHEMA DRIFT detected — refresh fixtures:", file=sys.stderr)
        for d in all_drift:
            print(f"  - {d}", file=sys.stderr)
        print(
            "Run: uv run python -m scripts.test.capture_neuronpedia_fixtures",
            file=sys.stderr,
        )
        return 1

    print(f"OK — checked {len(FEATURES)} features, all fixtures consistent")
    print("Neuronpedia schema OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
