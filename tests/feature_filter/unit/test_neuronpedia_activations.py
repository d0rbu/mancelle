# tests/feature_filter/unit/test_neuronpedia_activations.py
"""Fetch layer + cache tests for neuronpedia_activations."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import yaml

from errors import DataLoadError
from steering.eval.feature_filter.neuronpedia_activations import (
    ActivationEntry,
    FeatureActivationPayload,
    NeuronpediaActivationCache,
    fetch_feature_activations,
)
from steering.eval.feature_filter.types import InvalidHistogramError
from steering.eval.neuronpedia_client import (
    NeuronpediaAuthError,
    NeuronpediaFetchError,
    NeuronpediaNotPublishedError,
    NeuronpediaRateLimitError,
)
from steering.eval.neuronpedia_config import NeuronpediaConfig

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _mock_response(status: int, body: dict | str, headers: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    if isinstance(body, dict):
        resp.json.return_value = body
        resp.text = json.dumps(body)
    else:
        resp.json.side_effect = ValueError("not json")
        resp.text = body
    resp.headers = headers or {}
    return resp


def test_activation_entry_construction() -> None:
    e = ActivationEntry(
        tokens=["a", "b"],
        values=[0.0, 3.5],
        max_value=3.5,
        max_value_token_index=1,
        bin_min=0.0,
        bin_max=8.9,
        bin_contains=0.9998,
    )
    assert e.max_value == 3.5


def test_feature_activation_payload_construction() -> None:
    p = FeatureActivationPayload(
        feature_id=42,
        description="German articles",
        entries=[],
        freq_hist_bar_heights=[0.9, 0.1],
        freq_hist_bar_values=[0.0, 8.9],
        frac_nonzero=0.001,
        max_act_approx=44.6,
    )
    assert p.feature_id == 42


def test_fetch_happy_path(monkeypatch) -> None:
    body = json.loads((FIXTURES / "np_feature_minimal_happy.json").read_text())
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-testkey")

    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=_mock_response(200, body),
    ) as getter:
        payload = fetch_feature_activations("gemma-2-2b", "8-gemmascope-res-16k", 42)

    getter.assert_called_once_with(
        "https://neuronpedia.org/api/feature/gemma-2-2b/8-gemmascope-res-16k/42",
        headers={"X-Api-Key": "np-testkey", "Accept-Encoding": "gzip"},
        timeout=30.0,
    )
    assert payload.feature_id == 42
    assert payload.description == "German articles"
    assert payload.frac_nonzero == 0.001
    assert len(payload.entries) == 1
    assert payload.entries[0].max_value == 3.5
    assert payload.entries[0].bin_contains == 0.9998


def test_missing_api_key_halts_fast(monkeypatch) -> None:
    monkeypatch.delenv("NEURONPEDIA_API_KEY", raising=False)
    with pytest.raises(NeuronpediaAuthError):
        fetch_feature_activations("m", "s", 1)


def test_401_not_retried(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    call_count = 0

    def fake_get(*a, **kw):
        nonlocal call_count
        call_count += 1
        return _mock_response(401, "unauth")

    with patch("steering.eval.feature_filter.neuronpedia_activations.requests.get", side_effect=fake_get):
        with pytest.raises(NeuronpediaAuthError):
            fetch_feature_activations("m", "s", 1)
    assert call_count == 1


def test_429_retry_budget(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    cfg = NeuronpediaConfig(
        max_retries=2, backoff_base_seconds=1e-9, backoff_max_seconds=1e-9,
    )
    responses = [_mock_response(429, "busy", {"Retry-After": "0"}) for _ in range(3)]

    with patch("steering.eval.feature_filter.neuronpedia_activations.requests.get", side_effect=responses):
        with pytest.raises(NeuronpediaRateLimitError):
            fetch_feature_activations("m", "s", 1, config=cfg)


def test_5xx_retry_then_per_feature_error(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    cfg = NeuronpediaConfig(max_retries=1, backoff_base_seconds=1e-9, backoff_max_seconds=1e-9)
    responses = [_mock_response(503, "bad"), _mock_response(503, "bad")]
    with patch("steering.eval.feature_filter.neuronpedia_activations.requests.get", side_effect=responses):
        with pytest.raises(NeuronpediaFetchError):
            fetch_feature_activations("m", "s", 1, config=cfg)


def test_malformed_json(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=_mock_response(200, "not json"),
    ):
        with pytest.raises(NeuronpediaFetchError, match="Malformed JSON"):
            fetch_feature_activations("m", "s", 1)


def test_transient_request_exception_retried(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    cfg = NeuronpediaConfig(max_retries=2, backoff_base_seconds=1e-9, backoff_max_seconds=1e-9)
    body = json.loads((FIXTURES / "np_feature_minimal_happy.json").read_text())
    side_effects = [
        requests.exceptions.ChunkedEncodingError("closed"),
        _mock_response(200, body),
    ]
    with patch("steering.eval.feature_filter.neuronpedia_activations.requests.get", side_effect=side_effects):
        payload = fetch_feature_activations("m", "s", 42, config=cfg)
    assert payload.feature_id == 42


def test_nan_in_bin_fields_rejected(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    bad = {
        "explanations": [{"description": "x"}],
        "frac_nonzero": 0.1,
        "maxActApprox": 1.0,
        "freq_hist_data_bar_heights": [1.0],
        "freq_hist_data_bar_values": [0.0],
        "activations": [{
            "tokens": ["a"], "values": [0.0],
            "maxValue": float("nan"),
            "maxValueTokenIndex": 0,
            "binMin": 0.0, "binMax": 1.0, "binContains": 1.0,
        }],
    }
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=_mock_response(200, bad),
    ):
        with pytest.raises(NeuronpediaFetchError, match="NaN/inf"):
            fetch_feature_activations("m", "s", 1)


def test_values_tokens_length_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    bad = {
        "explanations": [{"description": "x"}],
        "frac_nonzero": 0.1,
        "maxActApprox": 1.0,
        "freq_hist_data_bar_heights": [1.0],
        "freq_hist_data_bar_values": [0.0],
        "activations": [{
            "tokens": ["a", "b"], "values": [0.0],
            "maxValue": 0.0,
            "maxValueTokenIndex": 0,
            "binMin": 0.0, "binMax": 1.0, "binContains": 1.0,
        }],
    }
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=_mock_response(200, bad),
    ):
        with pytest.raises(NeuronpediaFetchError, match="tokens .* != values"):
            fetch_feature_activations("m", "s", 1)


def test_histogram_length_mismatch_raises(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    bad = {
        "explanations": [{"description": "x"}],
        "frac_nonzero": 0.1,
        "maxActApprox": 1.0,
        "freq_hist_data_bar_heights": [1.0, 0.5],
        "freq_hist_data_bar_values": [0.0],
        "activations": [],
    }
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=_mock_response(200, bad),
    ):
        with pytest.raises(InvalidHistogramError):
            fetch_feature_activations("m", "s", 1)


def test_cache_init_writes_manifest(tmp_path) -> None:
    NeuronpediaActivationCache("gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "np_cache")
    assert (tmp_path / "np_cache" / "manifest.yaml").is_file()


def test_cache_init_rejects_mismatched_manifest(tmp_path) -> None:
    NeuronpediaActivationCache("gemma-2-2b", "source-a", tmp_path / "np_cache")
    with pytest.raises(DataLoadError):
        NeuronpediaActivationCache("gemma-2-2b", "source-b", tmp_path / "np_cache")


def test_cache_get_fetches_on_miss(tmp_path, monkeypatch) -> None:
    body = json.loads((FIXTURES / "np_feature_minimal_happy.json").read_text())
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    cache = NeuronpediaActivationCache("gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "np_cache")
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=_mock_response(200, body),
    ):
        payload = cache.get(42)
    assert payload.feature_id == 42
    assert (tmp_path / "np_cache" / "features" / "f_000042.json").is_file()


def test_cache_get_reads_on_hit(tmp_path, monkeypatch) -> None:
    body = json.loads((FIXTURES / "np_feature_minimal_happy.json").read_text())
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    cache = NeuronpediaActivationCache("gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "np_cache")

    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=_mock_response(200, body),
    ) as getter:
        cache.get(42)
        cache.get(42)
    assert getter.call_count == 1


def test_cache_atomic_write(tmp_path, monkeypatch) -> None:
    body = json.loads((FIXTURES / "np_feature_minimal_happy.json").read_text())
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    cache = NeuronpediaActivationCache("gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "np_cache")
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=_mock_response(200, body),
    ):
        cache.get(42)
    leftover = list((tmp_path / "np_cache" / "features").glob("*.tmp"))
    assert leftover == []


def test_cache_get_many_fetches_concurrently(tmp_path, monkeypatch) -> None:
    body = json.loads((FIXTURES / "np_feature_minimal_happy.json").read_text())
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    cache = NeuronpediaActivationCache("gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "np_cache")

    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=_mock_response(200, body),
    ):
        out = cache.get_many([1, 2, 3])
    assert set(out.keys()) == {1, 2, 3}


def test_cache_init_rejects_mismatched_schema_version(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "np_cache"
    NeuronpediaActivationCache("gemma-2-2b", "src", cache_dir)
    # Rewrite manifest with wrong schema_version
    manifest = cache_dir / "manifest.yaml"
    got = yaml.safe_load(manifest.read_text())
    got["schema_version"] = 999
    manifest.write_text(yaml.safe_dump(got))
    with pytest.raises(DataLoadError, match="schema_version"):
        NeuronpediaActivationCache("gemma-2-2b", "src", cache_dir)


def test_finite_float_none_rejected() -> None:
    from steering.eval.feature_filter.neuronpedia_activations import _finite_float
    with pytest.raises(NeuronpediaFetchError, match="is None"):
        _finite_float(None, "x", 1)


def test_finite_float_not_float_rejected() -> None:
    from steering.eval.feature_filter.neuronpedia_activations import _finite_float
    with pytest.raises(NeuronpediaFetchError, match="not a float"):
        _finite_float("not-a-num", "x", 1)


def test_parse_payload_not_dict() -> None:
    from steering.eval.feature_filter.neuronpedia_activations import _parse_payload
    with pytest.raises(NeuronpediaFetchError, match="not a dict"):
        _parse_payload([1, 2, 3], 1)


def test_parse_payload_activations_not_list() -> None:
    from steering.eval.feature_filter.neuronpedia_activations import _parse_payload
    with pytest.raises(NeuronpediaFetchError, match="activations.*not a list"):
        _parse_payload({"activations": "str"}, 1)


def test_parse_payload_skips_non_dict_activation_entry() -> None:
    from steering.eval.feature_filter.neuronpedia_activations import _parse_payload
    payload = {
        "activations": ["not-a-dict", {"tokens": ["a"], "values": [0.0], "maxValue": 0.0,
                                        "maxValueTokenIndex": 0, "binMin": 0.0,
                                        "binMax": 0.0, "binContains": 0.0}],
        "freq_hist_data_bar_heights": [1.0],
        "freq_hist_data_bar_values": [0.0],
        "frac_nonzero": 0.1,
        "maxActApprox": 1.0,
    }
    parsed = _parse_payload(payload, 1)
    assert len(parsed.entries) == 1


def test_parse_entry_empty_tokens_sets_idx_zero() -> None:
    from steering.eval.feature_filter.neuronpedia_activations import _parse_activation_entry
    entry = _parse_activation_entry({
        "tokens": [], "values": [], "maxValue": 0.0,
        "maxValueTokenIndex": 5, "binMin": 0.0, "binMax": 0.0, "binContains": 0.0,
    }, 1)
    assert entry.max_value_token_index == 0


def test_parse_payload_bar_heights_not_list() -> None:
    from steering.eval.feature_filter.neuronpedia_activations import _parse_payload
    with pytest.raises(InvalidHistogramError, match="not lists"):
        _parse_payload({
            "activations": [],
            "freq_hist_data_bar_heights": "x",
            "freq_hist_data_bar_values": [],
        }, 1)


def test_parse_payload_negative_bin_height() -> None:
    from steering.eval.feature_filter.neuronpedia_activations import _parse_payload
    with pytest.raises(InvalidHistogramError, match="negative bin"):
        _parse_payload({
            "activations": [],
            "freq_hist_data_bar_heights": [-0.1],
            "freq_hist_data_bar_values": [0.0],
            "frac_nonzero": 0.0, "maxActApprox": 0.0,
        }, 1)


def test_parse_payload_frac_nonzero_out_of_range() -> None:
    from steering.eval.feature_filter.neuronpedia_activations import _parse_payload
    with pytest.raises(NeuronpediaFetchError, match="outside"):
        _parse_payload({
            "activations": [],
            "freq_hist_data_bar_heights": [1.0],
            "freq_hist_data_bar_values": [0.0],
            "frac_nonzero": 2.0, "maxActApprox": 0.0,
        }, 1)


def test_fetch_network_error_exhausted_retries(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    cfg = NeuronpediaConfig(max_retries=1, backoff_base_seconds=1e-9, backoff_max_seconds=1e-9)
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        side_effect=requests.exceptions.ConnectTimeout("nope"),
    ):
        with pytest.raises(NeuronpediaFetchError, match="Network error"):
            fetch_feature_activations("m", "s", 1, config=cfg)


def test_fetch_none_status_raises(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    resp = MagicMock()
    resp.status_code = None
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=resp,
    ):
        with pytest.raises(NeuronpediaFetchError, match="no HTTP status"):
            fetch_feature_activations("m", "s", 1)


def test_fetch_unexpected_status_raises(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=_mock_response(418, "teapot"),
    ):
        with pytest.raises(NeuronpediaFetchError, match="returned 418"):
            fetch_feature_activations("m", "s", 1)


def test_get_many_all_cached(tmp_path, monkeypatch) -> None:
    body = json.loads((FIXTURES / "np_feature_minimal_happy.json").read_text())
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    cache = NeuronpediaActivationCache("gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "np_cache")
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        return_value=_mock_response(200, body),
    ):
        cache.get(1)
    # Second call: all cached, no fetch needed
    out = cache.get_many([1])
    assert set(out.keys()) == {1}


def test_cache_deletes_corrupt_file(tmp_path) -> None:
    cache_dir = tmp_path / "np_cache"
    cache = NeuronpediaActivationCache("gemma-2-2b", "src", cache_dir)
    feat_path = cache_dir / "features" / "f_000042.json"
    feat_path.write_text("not valid json {")
    result = cache.get_cached(42)
    assert result is None
    assert not feat_path.exists(), "corrupt cache file should be deleted"


def test_404_raises_not_published(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    resp = _mock_response(404, "SECRET-SERVER-BODY-should-never-leak")
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        side_effect=[resp],
    ):
        with pytest.raises(NeuronpediaNotPublishedError) as ei:
            fetch_feature_activations("gemma-2-2b", "20-gemmascope-2", 42)
    msg = str(ei.value)
    assert "42" in msg
    assert "SECRET-SERVER-BODY-should-never-leak" not in msg


def test_404_message_has_no_response_body(monkeypatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    resp = _mock_response(404, "<html>internal neuronpedia 404 page</html>")
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        side_effect=[resp],
    ):
        with pytest.raises(NeuronpediaNotPublishedError) as ei:
            fetch_feature_activations("m", "s", 7)
    text = str(ei.value)
    assert "<html>" not in text
    assert "internal neuronpedia" not in text


@pytest.mark.parametrize("status", [403, 410, 451, 418])
def test_non_404_4xx_is_fetch_error_not_not_published(status, monkeypatch) -> None:
    """The not_published gate is EXACTLY status == 404. Other 4xx must
    remain a genuine NeuronpediaFetchError (counts toward the error-rate
    abort); only a true 404 is the benign un-ingested-SAE signal."""
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    resp = _mock_response(status, "body")
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        side_effect=[resp],
    ):
        with pytest.raises(NeuronpediaFetchError) as ei:
            fetch_feature_activations("m", "s", 1)
    # And definitively NOT the not-published type.
    assert not isinstance(ei.value, NeuronpediaNotPublishedError)


def test_404_yields_not_published_not_fetch_error(monkeypatch) -> None:
    """Pin the ordering contract explicitly: a 404 yields
    NeuronpediaNotPublishedError, never the generic NeuronpediaFetchError
    (regression-locks the branch order documented in the source)."""
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "np-test")
    resp = _mock_response(404, "x")
    with patch(
        "steering.eval.feature_filter.neuronpedia_activations.requests.get",
        side_effect=[resp],
    ):
        with pytest.raises(NeuronpediaNotPublishedError):
            fetch_feature_activations("m", "s", 1)


def test_get_many_mixed_cache_hit_and_not_published(tmp_path, monkeypatch) -> None:
    """get_many with one cache-hit + one 404 miss: the hit is returned,
    the not-published miss is skipped, dict has only the hit."""
    from steering.eval.feature_filter.neuronpedia_activations import (
        NeuronpediaActivationCache,
    )

    cache = NeuronpediaActivationCache("m", "s", tmp_path)

    def _fake_get_cached(fid):
        return f"cached-{fid}" if fid == 1 else None

    def _fake_get(fid):
        if fid == 2:
            raise NeuronpediaNotPublishedError("np 404 for 2")
        return f"fetched-{fid}"

    monkeypatch.setattr(cache, "get_cached", _fake_get_cached)
    monkeypatch.setattr(cache, "get", _fake_get)
    out = cache.get_many([1, 2])
    assert out == {1: "cached-1"}


def test_get_many_skips_not_published_returns_others(tmp_path, monkeypatch) -> None:
    cache = NeuronpediaActivationCache("m", "s", tmp_path)

    def _fake_get(fid):
        if fid == 2:
            raise NeuronpediaNotPublishedError("np 404 for feature 2")
        return f"payload-{fid}"

    monkeypatch.setattr(cache, "get", _fake_get)
    out = cache.get_many([1, 2, 3])
    assert set(out) == {1, 3}
    assert 2 not in out


def test_get_many_propagates_other_exceptions(tmp_path, monkeypatch) -> None:
    cache = NeuronpediaActivationCache("m", "s", tmp_path)

    def _fake_get(fid):
        if fid == 2:
            raise NeuronpediaFetchError("real transient failure")
        return f"payload-{fid}"

    monkeypatch.setattr(cache, "get", _fake_get)
    with pytest.raises(NeuronpediaFetchError):
        cache.get_many([1, 2, 3])
