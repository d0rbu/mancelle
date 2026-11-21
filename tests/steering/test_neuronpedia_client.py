"""Unit tests for :func:`fetch_feature_description` and :class:`NeuronpediaClient`.

This file covers the synchronous, non-concurrent path. Concurrent batch
behavior, retry/backoff, and cache migration live in
``test_neuronpedia_client_concurrent.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.neuronpedia_client import (
    NEURONPEDIA_API_KEY_ENV,
    NeuronpediaAuthError,
    NeuronpediaClient,
    NeuronpediaEmptyDescriptionError,
    NeuronpediaFetchError,
    NeuronpediaNotPublishedError,
    NeuronpediaSourceMismatchError,
    fetch_feature_description,
)
from steering.eval.neuronpedia_config import NeuronpediaConfig

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any | None = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}
        self.text = text or (str(json_data) if json_data is not None else "")

    def json(self) -> dict[str, Any]:
        if self._json_data is None:
            raise ValueError("no json data")
        return self._json_data


def _ok_response(description: str = "formal writing") -> _FakeResponse:
    return _FakeResponse(
        200,
        json_data={
            "modelId": "gemma-2-2b",
            "layer": "8-gemmascope-res-16k",
            "index": "42",
            "explanations": [{"description": description}],
        },
    )


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NEURONPEDIA_API_KEY_ENV, "np-test-key")


@pytest.fixture
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip backoff sleeps so tests don't take real wall time."""
    import steering.eval.neuronpedia_client as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)


# ---------------------------------------------------------------------------
# fetch_feature_description
# ---------------------------------------------------------------------------


class TestFetchFeatureDescription:
    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_returns_first_description(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _ok_response("formal writing")
        desc = fetch_feature_description("gemma-2-2b", "8-gemmascope-res-16k", 42)
        assert desc == "formal writing"

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_calls_correct_url(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _ok_response()
        fetch_feature_description("gemma-2-2b", "8-gemmascope-res-16k", 42)
        url = mock_get.call_args.args[0]
        assert "neuronpedia.org/api/feature/gemma-2-2b/8-gemmascope-res-16k/42" in url

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_sends_api_key_header(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _ok_response()
        fetch_feature_description("gemma-2-2b", "8-gemmascope-res-16k", 42)
        headers = mock_get.call_args.kwargs["headers"]
        assert headers["X-Api-Key"] == "np-test-key"

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_sends_timeout(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _ok_response()
        fetch_feature_description("gemma-2-2b", "8-gemmascope-res-16k", 42)
        assert mock_get.call_args.kwargs["timeout"] == 30.0

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_no_explanations_raises(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _FakeResponse(
            200,
            json_data={
                "modelId": "gemma-2-2b",
                "layer": "8-gemmascope-res-16k",
                "index": "99",
                "explanations": [],
            },
        )
        with pytest.raises(NeuronpediaFetchError, match="No explanations"):
            fetch_feature_description("gemma-2-2b", "8-gemmascope-res-16k", 99)

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_no_description_in_explanation_raises(
        self, mock_get: MagicMock,
    ) -> None:
        mock_get.return_value = _FakeResponse(
            200,
            json_data={
                "modelId": "gemma-2-2b",
                "layer": "8-gemmascope-res-16k",
                "index": "99",
                "explanations": [{"author": "x"}],
            },
        )
        with pytest.raises(NeuronpediaFetchError, match="No description"):
            fetch_feature_description("gemma-2-2b", "8-gemmascope-res-16k", 99)

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_non_string_description_raises(
        self, mock_get: MagicMock,
    ) -> None:
        """If the API returns a description that isn't a string (e.g. a
        list), the parser must raise rather than silently coercing via
        ``str()``."""
        mock_get.return_value = _FakeResponse(
            200,
            json_data={
                "modelId": "gemma-2-2b",
                "layer": "8-gemmascope-res-16k",
                "index": "99",
                "explanations": [{"description": ["list", "of", "strings"]}],
            },
        )
        with pytest.raises(NeuronpediaFetchError, match="unexpected type"):
            fetch_feature_description("gemma-2-2b", "8-gemmascope-res-16k", 99)

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_non_dict_payload_raises(
        self, mock_get: MagicMock,
    ) -> None:
        """If the API returns something that isn't a dict (e.g. a list),
        the parser raises a clear NeuronpediaFetchError."""
        mock_get.return_value = _FakeResponse(
            200, json_data=["not", "a", "dict"],  # type: ignore[arg-type]
        )
        with pytest.raises(NeuronpediaFetchError, match="not a dict"):
            fetch_feature_description("gemma-2-2b", "8-gemmascope-res-16k", 1)

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_401_raises_auth_error(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _FakeResponse(401, text="Unauthorized")
        with pytest.raises(NeuronpediaAuthError):
            fetch_feature_description("gemma-2-2b", "8-gemmascope-res-16k", 42)

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_404_raises_not_published(self, mock_get: MagicMock) -> None:
        # Task 29: a 404 now maps to the typed
        # ``NeuronpediaNotPublishedError`` (the expected state for SAE
        # releases Neuronpedia hasn't ingested yet, e.g. gemma-scope-2)
        # rather than the generic ``NeuronpediaFetchError``. The typed
        # error is a ``RuntimeError`` subclass and is NOT a
        # ``NeuronpediaError``, so it is not silently swallowed by the
        # batch path's ``except NeuronpediaError`` skip.
        mock_get.return_value = _FakeResponse(404, text="Not Found")
        with pytest.raises(NeuronpediaNotPublishedError):
            fetch_feature_description("gemma-2-2b", "8-gemmascope-res-16k", 42)
        assert not issubclass(NeuronpediaNotPublishedError, NeuronpediaFetchError)


# ---------------------------------------------------------------------------
# NeuronpediaClient — caching basics
# ---------------------------------------------------------------------------


class TestNeuronpediaClient:
    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_caches_to_disk(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        mock_get.return_value = _ok_response("test desc")
        client = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "cache.yaml",
        )
        assert client.get_description(42) == "test desc"
        assert mock_get.call_count == 1
        # Second call hits cache, not API.
        assert client.get_description(42) == "test desc"
        assert mock_get.call_count == 1

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_cache_persists_across_instances(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        mock_get.return_value = _ok_response("persisted")
        cache_path = tmp_path / "cache.yaml"
        c1 = NeuronpediaClient("gemma-2-2b", "8-gemmascope-res-16k", cache_path)
        c1.get_description(10)
        c2 = NeuronpediaClient("gemma-2-2b", "8-gemmascope-res-16k", cache_path)
        assert c2.get_description(10) == "persisted"
        assert mock_get.call_count == 1

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_batch_fetch_returns_all(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        mock_get.return_value = _ok_response("batch desc")
        client = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "cache.yaml",
        )
        descs = client.get_descriptions([1, 2, 3])
        assert descs == {1: "batch desc", 2: "batch desc", 3: "batch desc"}
        assert mock_get.call_count == 3

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_get_descriptions_cache_only_returns_cached_without_http(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        mock_get.return_value = _ok_response("cached desc")
        client = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "cache.yaml",
        )
        # Prime the cache for feature 42 (one HTTP call).
        client.get_description(42)
        assert mock_get.call_count == 1
        # cache_only request including an UNCACHED id (99) must return only the
        # cached entry and make ZERO further HTTP calls.
        descs = client.get_descriptions([42, 99], cache_only=True)
        assert descs == {42: "cached desc"}
        assert mock_get.call_count == 1

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_get_descriptions_cache_only_empty_cache_returns_empty_no_http(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        client = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "cache.yaml",
        )
        descs = client.get_descriptions([1, 2, 3], cache_only=True)
        assert descs == {}
        assert mock_get.call_count == 0

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_get_descriptions_default_still_fetches_uncached(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # Regression: default (cache_only=False) behavior is unchanged.
        mock_get.return_value = _ok_response("fetched")
        client = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "cache.yaml",
        )
        descs = client.get_descriptions([7, 8])
        assert descs == {7: "fetched", 8: "fetched"}
        assert mock_get.call_count == 2

    def test_get_cached_returns_none_when_missing(self, tmp_path: Path) -> None:
        client = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "cache.yaml",
        )
        assert client.get_cached(99) is None

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_get_cached_returns_value_when_present(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        mock_get.return_value = _ok_response("cached value")
        client = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "cache.yaml",
        )
        client.get_description(42)
        assert client.get_cached(42) == "cached value"

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_get_descriptions_emits_profiler_cache_counters(
        self,
        mock_get: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_get.return_value = _ok_response("profiled")
        client = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "cache.yaml",
        )

        with AxBenchProfiler(tmp_path / "profile", synchronize_cuda=False):
            descs = client.get_descriptions([1, 2])

        assert descs == {1: "profiled", 2: "profiled"}
        events = [
            json.loads(line)
            for line in (tmp_path / "profile" / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        counters = {
            event["name"]: event["value"]
            for event in events
            if event.get("type") == "counter"
        }
        assert counters["neuronpedia.descriptions_requested"] == 2.0
        assert counters["neuronpedia.descriptions_cached"] == 0.0
        assert counters["neuronpedia.descriptions_to_fetch"] == 2.0
        assert counters["neuronpedia.descriptions_fetched"] == 2.0
        assert counters["neuronpedia.descriptions_failed"] == 0.0


class TestMissingApiKey:
    def test_raises_on_first_fetch_when_env_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.delenv(NEURONPEDIA_API_KEY_ENV, raising=False)
        client = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "cache.yaml",
        )
        with pytest.raises(NeuronpediaAuthError, match=NEURONPEDIA_API_KEY_ENV):
            client.get_description(42)

    def test_raises_with_empty_string_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Empty string is treated as unset (matches OPENROUTER pattern)."""
        monkeypatch.setenv(NEURONPEDIA_API_KEY_ENV, "")
        client = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", tmp_path / "cache.yaml",
        )
        with pytest.raises(NeuronpediaAuthError, match=NEURONPEDIA_API_KEY_ENV):
            client.get_description(42)


# ---------------------------------------------------------------------------
# Source/SAE mismatch guard (opt-in) — issue #657 follow-up
# ---------------------------------------------------------------------------


def _feature_id_from_url(url: str) -> int:
    return int(url.rsplit("/", 1)[-1])


def _published_only(published: set[int]):
    """Build a ``requests.get`` side effect: 200 with a description for
    ``published`` feature ids, 404 (Not Found) for every other id."""

    def _side_effect(url: str, *, headers: Any, timeout: Any) -> _FakeResponse:
        fid = _feature_id_from_url(url)
        if fid in published:
            return _FakeResponse(
                200,
                json_data={
                    "modelId": "gemma-2-2b",
                    "layer": "8-gemmascope-res-16k",
                    "index": str(fid),
                    "explanations": [{"description": f"desc-{fid}"}],
                },
            )
        return _FakeResponse(404, text="Not Found")

    return _side_effect


def _empty_except(published: set[int]):
    """Build a ``requests.get`` side effect: 200 with a description for
    ``published`` ids, 200 with EMPTY explanations (not 404) for the rest --
    the shape a source uses when it hosts the index but has no description."""

    def _side_effect(url: str, *, headers: Any, timeout: Any) -> _FakeResponse:
        fid = _feature_id_from_url(url)
        explanations = (
            [{"description": f"desc-{fid}"}] if fid in published else []
        )
        return _FakeResponse(
            200,
            json_data={
                "modelId": "gemma-2-2b",
                "layer": "8-gemmascope-res-16k",
                "index": str(fid),
                "explanations": explanations,
            },
        )

    return _side_effect


def _server_error_except(published: set[int]):
    """Build a ``requests.get`` side effect: 200 with a description for
    ``published`` ids, HTTP 500 for the rest -- the shape the live API uses for
    a wholly nonexistent / typo'd source (it 5xxs, not 404s -- see
    reference_neuronpedia_live_404). Every non-published id retries the backoff
    budget and finally raises a transient ``NeuronpediaFetchError``."""

    def _side_effect(url: str, *, headers: Any, timeout: Any) -> _FakeResponse:
        fid = _feature_id_from_url(url)
        if fid in published:
            return _FakeResponse(
                200,
                json_data={
                    "modelId": "gemma-2-2b",
                    "layer": "8-gemmascope-res-16k",
                    "index": str(fid),
                    "explanations": [{"description": f"desc-{fid}"}],
                },
            )
        return _FakeResponse(500, text="Internal Server Error")

    return _side_effect


def _all_401(url: str, *, headers: Any, timeout: Any) -> _FakeResponse:
    """A ``requests.get`` side effect that returns HTTP 401 for every request --
    the shape of a present-but-rejected/invalid API key (each request raises
    NeuronpediaAuthError)."""
    del url, headers, timeout
    return _FakeResponse(401, text="invalid api key")


class TestSourceMismatchGuard:
    """``get_descriptions`` opt-in not-published-rate guard (issue #657).

    A high 404 / not-published rate signals a Neuronpedia source that does
    not match the SAE being queried (e.g. a 16k source against a 65k SAE).
    Callers for whom the description is load-bearing (AxBench-Base Stage 0a
    concept selection) set a positive ``max_not_published_rate`` and get a
    loud failure instead of a silently truncated / mislabeled concept pool.
    Callers for whom the description is decorative (autointerp) leave it at
    the ``0.0`` default (guard disabled) and are unaffected (issue #691).
    """

    def _client(
        self, tmp_path: Path, config: NeuronpediaConfig | None = None,
    ) -> NeuronpediaClient:
        return NeuronpediaClient(
            "gemma-2-2b",
            "8-gemmascope-res-16k",
            tmp_path / "cache.yaml",
            config=config or NeuronpediaConfig(max_concurrent=4),
        )

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_mass_not_published_raises_when_opted_in(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # 3 of 4 requested features 404 (75% > 0.5) -> loud failure that
        # names the configured source and model.
        mock_get.side_effect = _published_only({0})
        client = self._client(tmp_path)
        with pytest.raises(NeuronpediaSourceMismatchError) as excinfo:
            client.get_descriptions([0, 1, 2, 3], max_not_published_rate=0.5)
        msg = str(excinfo.value)
        assert "8-gemmascope-res-16k" in msg
        assert "gemma-2-2b" in msg

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_low_not_published_rate_drops_without_raising(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # 1 of 4 404 (25% < 0.5) -> no raise; the 404 feature is omitted.
        mock_get.side_effect = _published_only({0, 1, 2})
        client = self._client(tmp_path)
        out = client.get_descriptions(
            [0, 1, 2, 3], max_not_published_rate=0.5,
        )
        assert set(out) == {0, 1, 2}

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_guard_disabled_by_default_tolerates_all_not_published(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # The default rate 0.0 disables the guard -> preserves the non-fatal
        # drop contract even at a 100% not-published rate (decorative callers
        # / issue #691 stay byte-identical).
        mock_get.side_effect = _published_only(set())
        client = self._client(tmp_path)
        out = client.get_descriptions([0, 1, 2, 3])
        assert out == {}

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_empty_explanations_do_not_trip_the_not_published_guard(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # A published feature with an EMPTY explanations payload raises
        # NeuronpediaFetchError (not a 404). That is the #691 "decorative
        # miss" case and must NOT count toward the source-mismatch guard,
        # even at a 100% empty rate.
        def _all_empty(url: str, *, headers: Any, timeout: Any) -> _FakeResponse:
            fid = _feature_id_from_url(url)
            return _FakeResponse(
                200,
                json_data={
                    "modelId": "gemma-2-2b",
                    "layer": "8-gemmascope-res-16k",
                    "index": str(fid),
                    "explanations": [],
                },
            )

        mock_get.side_effect = _all_empty
        client = self._client(tmp_path)
        out = client.get_descriptions(
            [0, 1, 2, 3], max_not_published_rate=0.5,
        )
        assert out == {}

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_rate_is_over_total_requested_not_over_uncached_attempts(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # Warm-cache analogue: 8 features succeed and cache; a later request
        # for all 10 leaves only fids 8,9 to fetch, and both 404. 2/10 = 20%
        # of the REQUESTED population < 0.5 -> must NOT raise, even though
        # 2/2 = 100% of this call's fetch ATTEMPTS were 404.
        client = self._client(tmp_path)
        mock_get.side_effect = _published_only(set(range(8)))
        warm = client.get_descriptions(list(range(8)))
        assert set(warm) == set(range(8))
        out = client.get_descriptions(
            list(range(10)), max_not_published_rate=0.5,
        )
        assert set(out) == set(range(8))

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_cache_only_short_circuits_before_guard(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # cache_only returns before any fetch, so an ACTIVE guard is never
        # evaluated (no network, no 404s) even with an empty cache.
        client = self._client(tmp_path)
        out = client.get_descriptions(
            [0, 1, 2, 3], cache_only=True, max_not_published_rate=0.01,
        )
        assert out == {}
        mock_get.assert_not_called()

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_mismatch_persists_successful_fetches_before_raising(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # On a guard raise the successfully-fetched descriptions ARE persisted
        # (namespaced by (model, source)) so a later run that raises the
        # threshold reuses them instead of a cold re-fetch; the partial cache is
        # valid only for its own (here wrong) namespace, so it never pollutes a
        # corrected run.
        cache_path = tmp_path / "cache.yaml"
        mock_get.side_effect = _published_only({0})  # fid 0 ok, 1-3 404 (75%)
        client = NeuronpediaClient(
            "gemma-2-2b", "8-gemmascope-res-16k", cache_path,
            config=NeuronpediaConfig(max_concurrent=4),
        )
        with pytest.raises(NeuronpediaSourceMismatchError):
            client.get_descriptions([0, 1, 2, 3], max_not_published_rate=0.5)
        assert cache_path.exists()
        assert client.get_cached(0) == "desc-0"

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_not_published_counter_recorded_on_fully_cached_early_return(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # descriptions_not_published is emitted on the n_to_fetch==0
        # early-return path, not only the main fetch path.
        mock_get.side_effect = _published_only({1, 2})
        client = self._client(tmp_path)
        client.get_descriptions([1, 2])  # warm the cache
        with AxBenchProfiler(tmp_path / "profile", synchronize_cuda=False):
            client.get_descriptions([1, 2])  # all cached -> early return
        events = [
            json.loads(line)
            for line in (tmp_path / "profile" / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        counters = {
            e["name"]: e["value"] for e in events if e.get("type") == "counter"
        }
        assert counters["neuronpedia.descriptions_to_fetch"] == 0.0
        assert counters["neuronpedia.descriptions_not_published"] == 0.0

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_empty_explanations_raise_typed_error(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # A 200 with empty explanations is a distinct, typed error that still
        # IS a NeuronpediaFetchError (so #691's decorative `except
        # NeuronpediaFetchError` keeps treating it as a non-fatal drop).
        mock_get.side_effect = _empty_except(set())
        with pytest.raises(NeuronpediaEmptyDescriptionError):
            fetch_feature_description("gemma-2-2b", "8-gemmascope-res-16k", 3)
        assert issubclass(NeuronpediaEmptyDescriptionError, NeuronpediaFetchError)

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_mass_empty_explanations_trips_missing_guard(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # A source that answers 200-empty (not 404) for most indices trips the
        # missing-rate guard, even though the 404 rate is 0.
        mock_get.side_effect = _empty_except({0})  # 3/4 empty = 75% > 0.5
        client = self._client(tmp_path)
        with pytest.raises(NeuronpediaSourceMismatchError, match="empty"):
            client.get_descriptions(
                [0, 1, 2, 3], max_missing_rate=0.5, min_mismatch_features=1,
            )

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_floor_prevents_404_small_pool_false_positive(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # 1/4 = 25% > 0.02 rate, but below the absolute floor (5) -> no raise.
        mock_get.side_effect = _published_only({0, 1, 2})
        client = self._client(tmp_path)
        out = client.get_descriptions(
            [0, 1, 2, 3], max_not_published_rate=0.02, min_mismatch_features=5,
        )
        assert set(out) == {0, 1, 2}

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_floor_prevents_missing_small_pool_false_positive(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # 3/4 empty = 75% > 0.5 missing rate, but below the floor (5) -> no raise.
        mock_get.side_effect = _empty_except({0})
        client = self._client(tmp_path)
        out = client.get_descriptions(
            [0, 1, 2, 3], max_missing_rate=0.5, min_mismatch_features=5,
        )
        assert set(out) == {0}

    def _no_retry_client(self, tmp_path: Path) -> NeuronpediaClient:
        # max_retries=0 so a 5xx raises immediately (no backoff sleeps in test).
        return self._client(
            tmp_path, config=NeuronpediaConfig(max_concurrent=4, max_retries=0),
        )

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_mass_transient_failure_trips_total_failure_guard(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # A wholly nonexistent / typo'd source 5xxs every request (not 404):
        # those are transient failures that trip NEITHER the 404 nor the
        # missing guard (both stay 0). The total-failure guard catches them so
        # the misconfig fails loud instead of misdiagnosing as a network blip.
        mock_get.side_effect = _server_error_except(set())
        client = self._no_retry_client(tmp_path)
        with pytest.raises(NeuronpediaSourceMismatchError) as excinfo:
            client.get_descriptions(
                list(range(10)),
                max_not_published_rate=0.02,
                max_missing_rate=0.5,
                max_failed_rate=0.5,
                min_mismatch_features=5,
            )
        msg = str(excinfo.value)
        assert "8-gemmascope-res-16k" in msg
        assert "gemma-2-2b" in msg

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_total_failure_guard_disabled_by_default(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # max_failed_rate defaults to 0.0 (disabled) -> a 100% transient-failure
        # run drops silently (decorative callers / issue #691 unaffected), even
        # though the 404 and missing guards are opted in (they see 0 of each).
        mock_get.side_effect = _server_error_except(set())
        client = self._no_retry_client(tmp_path)
        out = client.get_descriptions(
            list(range(10)),
            max_not_published_rate=0.02,
            max_missing_rate=0.5,
            min_mismatch_features=1,
        )
        assert out == {}

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_floor_prevents_total_failure_small_pool_false_positive(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # 4/4 = 100% transient failure, but below the absolute floor (5) -> no
        # raise: a brief network blip on a tiny pool must not read as a mismatch.
        mock_get.side_effect = _server_error_except(set())
        client = self._no_retry_client(tmp_path)
        out = client.get_descriptions(
            [0, 1, 2, 3], max_failed_rate=0.5, min_mismatch_features=5,
        )
        assert out == {}

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_total_failure_guard_counts_all_drop_kinds(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # The total-failure guard aggregates 404 + empty + transient: a source
        # with a mix that trips none of them individually (below their rates)
        # can still be caught when the COMBINED drop rate is mass.
        # 8 of 10 5xx (transient) -> 404 rate 0, missing rate 0, but failed
        # rate 0.8 > 0.5 -> raise.
        mock_get.side_effect = _server_error_except({0, 1})
        client = self._no_retry_client(tmp_path)
        with pytest.raises(NeuronpediaSourceMismatchError) as excinfo:
            client.get_descriptions(
                list(range(10)),
                max_not_published_rate=0.02,
                max_missing_rate=0.5,
                max_failed_rate=0.5,
                min_mismatch_features=5,
            )
        assert "80.0%" in str(excinfo.value) or "8/10" in str(excinfo.value)

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_invalid_key_raises_auth_not_source_mismatch(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        # A present-but-rejected key 401s every request. Auth errors are counted
        # apart (n_auth), NOT toward the failed rate, so a guarded caller gets a
        # NeuronpediaAuthError naming the key -- not a NeuronpediaSourceMismatchError
        # blaming the source. (The autouse fixture sets a non-empty key, so the
        # up-front missing-key check passes and the 401s surface in-loop.)
        mock_get.side_effect = _all_401
        client = self._no_retry_client(tmp_path)
        with pytest.raises(NeuronpediaAuthError, match=NEURONPEDIA_API_KEY_ENV):
            client.get_descriptions(
                list(range(10)),
                max_not_published_rate=0.02,
                max_missing_rate=0.5,
                max_failed_rate=0.5,
                min_mismatch_features=1,
            )

    def test_missing_key_with_guard_active_fails_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A guarded caller (Stage 0a) with NO key fails fast up front with the
        # auth error, before any request -- not after tripping max_failed_rate.
        monkeypatch.delenv(NEURONPEDIA_API_KEY_ENV, raising=False)
        client = self._client(tmp_path)
        with pytest.raises(NeuronpediaAuthError, match=NEURONPEDIA_API_KEY_ENV):
            client.get_descriptions([0, 1, 2, 3], max_not_published_rate=0.02)

    @patch("steering.eval.neuronpedia_client.requests.get")
    def test_decorative_caller_missing_key_returns_cached_subset(
        self, mock_get: MagicMock, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A decorative caller (all rates 0.0, issue #691) with NO key and a
        # partial cache keeps the graceful-drop contract: the up-front auth
        # check is skipped (no guard), the cached subset is returned, and the
        # uncached ids drop on their per-request auth error (no raise).
        monkeypatch.delenv(NEURONPEDIA_API_KEY_ENV, raising=False)
        mock_get.side_effect = _all_401
        client = self._client(tmp_path)
        with client._cache_lock:
            client._cache[0] = "cached-desc-0"
        out = client.get_descriptions([0, 1, 2])
        assert out == {0: "cached-desc-0"}
