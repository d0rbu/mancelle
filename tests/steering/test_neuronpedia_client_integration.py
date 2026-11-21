"""Live integration test against the real Neuronpedia API.

Skipped when ``NEURONPEDIA_API_KEY`` is not set so vanilla
``uv run pytest`` (no creds) stays green. Run locally / CI with::

    NEURONPEDIA_API_KEY=np-... uv run pytest tests/test_neuronpedia_client_integration.py -v -s

Tests are kept conservative on the live wire: small fid range, partial
success accepted, generous retry budget, low concurrency to avoid 429s.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from steering.eval.neuronpedia_client import (
    NEURONPEDIA_API_KEY_ENV,
    NeuronpediaClient,
)
from steering.eval.neuronpedia_config import NeuronpediaConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get(NEURONPEDIA_API_KEY_ENV),
    reason=f"{NEURONPEDIA_API_KEY_ENV} not set; skipping live API test",
)


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "live_cache.yaml"


def test_fetch_a_known_feature(cache_path: Path) -> None:
    """Fetch a single feature description live."""
    client = NeuronpediaClient(
        "gemma-2-2b", "8-gemmascope-res-16k", cache_path,
        config=NeuronpediaConfig(max_concurrent=1, max_retries=2),
    )
    desc = client.get_description(0)
    assert isinstance(desc, str)
    assert len(desc) >= 1


def test_batch_fetch_with_concurrency(cache_path: Path) -> None:
    """Fetch ~20 features concurrently and verify shape."""
    client = NeuronpediaClient(
        "gemma-2-2b", "8-gemmascope-res-16k", cache_path,
        config=NeuronpediaConfig(max_concurrent=5, max_retries=3),
    )
    fids = list(range(20))
    descs = client.get_descriptions(fids)
    # Some features may not have explanations -- accept partial success
    # but require >= half so a flaky network doesn't false-fail.
    assert len(descs) >= len(fids) // 2
    for fid, desc in descs.items():
        assert fid in fids
        assert isinstance(desc, str)
        assert len(desc) >= 1


def test_cache_warm_path_zero_calls(cache_path: Path) -> None:
    """A second client opened on the same cache must not hit the wire."""
    client = NeuronpediaClient(
        "gemma-2-2b", "8-gemmascope-res-16k", cache_path,
        config=NeuronpediaConfig(max_concurrent=2, max_retries=2),
    )
    fids = [0, 1, 2]
    first = client.get_descriptions(fids)

    cached_fids = list(first.keys())
    assert cached_fids, (
        "Live API returned 0 descriptions for fids [0, 1, 2]; cannot "
        "exercise warm-cache path. Either Neuronpedia is down, the key "
        "is wrong, or the target SAE stopped exposing low-index features. "
        "Investigate before trusting this test."
    )

    with patch("steering.eval.neuronpedia_client.requests.get") as mget:
        client2 = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", cache_path,
            config=NeuronpediaConfig(max_concurrent=2, max_retries=2),
        )
        second = client2.get_descriptions(cached_fids)
        assert mget.call_count == 0
        assert second == first
