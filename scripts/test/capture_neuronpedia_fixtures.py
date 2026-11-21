"""One-shot fixture capture: fetches two known features from Neuronpedia
and stores the trimmed payloads under tests/feature_filter/fixtures/.

Usage:
    NEURONPEDIA_API_KEY=... uv run python -m scripts.test.capture_neuronpedia_fixtures
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from steering.eval.feature_filter.neuronpedia_activations import (
    fetch_feature_activations,
)

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "feature_filter" / "fixtures"


def _dump(model_id: str, sae_source: str, feature_id: int, name: str) -> None:
    payload = fetch_feature_activations(model_id, sae_source, feature_id)
    body = {
        "feature_id": payload.feature_id,
        "description": payload.description,
        "entries": [asdict(e) for e in payload.entries],
        "freq_hist_bar_heights": payload.freq_hist_bar_heights,
        "freq_hist_bar_values": payload.freq_hist_bar_values,
        "frac_nonzero": payload.frac_nonzero,
        "max_act_approx": payload.max_act_approx,
    }
    out = FIXTURES / name
    out.write_text(json.dumps(body, indent=2))
    print(f"wrote {out}")


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    _dump("gemma-2-2b", "8-gemmascope-res-16k", 42, "np_feature_gemma_res16k_42.json")
    _dump("gemma-2-2b", "8-gemmascope-res-16k", 100, "np_feature_gemma_res16k_100.json")


if __name__ == "__main__":
    main()
