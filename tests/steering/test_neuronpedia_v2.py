"""Tests for Neuronpedia v2 grammar (gemma-scope-2)."""

from unittest.mock import MagicMock, patch

import pytest

from steering.eval.neuronpedia_client import (
    NeuronpediaClient,
    NeuronpediaFetchError,
    NeuronpediaNotPublishedError,
    fetch_feature_description,
)
from steering.eval.neuronpedia_config import NeuronpediaConfig
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    parse_neuronpedia_source,
    resolve_neuronpedia_ids,
)


class TestNeuronpediaNotPublishedError:
    def test_error_type_exists(self):
        e = NeuronpediaNotPublishedError("not found")
        assert isinstance(e, Exception)
        assert "not found" in str(e)


class TestResolveNeuronpediaIdsV2:
    def test_v2_source_format(self):
        model_id, sae_source = resolve_neuronpedia_ids(
            model="gemma-3-4b-it",
            release=SAERelease.gemmascope_2,
            activation_type=ActivationType.residual,
            layer=22,
            width=SAEWidth.W_16K,
            l0=L0Tier.medium,
        )
        assert model_id == "gemma-3-4b-it"
        assert sae_source == "22-gemmascope-2-res-16k"

    def test_v2_validation_propagates(self):
        with pytest.raises(ValueError, match=r"Layer 10"):
            resolve_neuronpedia_ids(
                model="gemma-3-4b-it",
                release=SAERelease.gemmascope_2,
                activation_type=ActivationType.residual,
                layer=10,  # invalid
                width=SAEWidth.W_16K,
                l0=L0Tier.medium,
            )

    def test_v1_unchanged(self):
        model_id, sae_source = resolve_neuronpedia_ids(
            model="gemma-2-2b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=8,
            width=SAEWidth.W_16K,
        )
        assert (model_id, sae_source) == ("gemma-2-2b", "8-gemmascope-res-16k")


class TestParseNeuronpediaSourceV2:
    def test_v2_round_trip(self):
        release, activation, layer, width, l0_tier = parse_neuronpedia_source(
            "22-gemmascope-2-res-16k"
        )
        assert release == SAERelease.gemmascope_2
        assert activation == ActivationType.residual
        assert layer == 22
        assert width == SAEWidth.W_16K
        assert l0_tier == L0Tier.medium

    def test_v2_explicit_tier_round_trip(self):
        release, activation, layer, width, l0_tier = parse_neuronpedia_source(
            "22-gemmascope-2-res-16k-medium"
        )
        assert release == SAERelease.gemmascope_2
        assert activation == ActivationType.residual
        assert layer == 22
        assert width == SAEWidth.W_16K
        assert l0_tier == L0Tier.medium

    def test_v1_round_trip_returns_none_l0_tier(self):
        result = parse_neuronpedia_source("8-gemmascope-res-16k")
        assert result[0] == SAERelease.gemmascope
        assert result[1] == ActivationType.residual
        assert result[2] == 8
        assert result[3] == SAEWidth.W_16K
        assert result[4] is None

    def test_v2_invalid_tier_raises(self):
        with pytest.raises(ValueError, match=r"l0 tier"):
            parse_neuronpedia_source("22-gemmascope-2-res-16k-huge")

    def test_wrong_segment_count_raises(self):
        with pytest.raises(ValueError, match=r"segments"):
            parse_neuronpedia_source("22-gemmascope-2-res")


class TestNeuronpediaClient404Handling:
    """The real client function is ``fetch_feature_description`` (there is
    no ``fetch_feature_activations`` symbol). It performs the HTTP call via
    the module-global ``requests.get``, reads ``resp.status_code`` /
    ``resp.text``, and dispatches
    on status. A 404 must raise the typed
    :class:`NeuronpediaNotPublishedError`; other non-2xx must still raise a
    generic ``NeuronpediaError``. ``max_retries=0`` keeps the 5xx case fast
    and deterministic (no real backoff sleeps)."""

    def test_404_raises_not_published(self):
        resp = MagicMock(status_code=404, text="Not found")
        with patch.dict("os.environ", {"NEURONPEDIA_API_KEY": "np-test"}):
            with patch("requests.get", return_value=resp):
                with pytest.raises(NeuronpediaNotPublishedError):
                    fetch_feature_description(
                        "gemma-3-4b-it",
                        "22-gemmascope-2-res-16k",
                        42,
                        config=NeuronpediaConfig(max_retries=0),
                    )

    def test_500_still_raises_generic_runtime(self):
        resp = MagicMock(status_code=500, text="oops")
        with patch.dict("os.environ", {"NEURONPEDIA_API_KEY": "np-test"}):
            with patch("requests.get", return_value=resp):
                with pytest.raises(Exception) as excinfo:
                    fetch_feature_description(
                        "gemma-3-4b-it",
                        "22-gemmascope-2-res-16k",
                        42,
                        config=NeuronpediaConfig(max_retries=0),
                    )
        # The 500 path must NOT be reclassified as the typed
        # not-published error — only 404 gets that treatment.
        assert not isinstance(excinfo.value, NeuronpediaNotPublishedError)


class TestBatchToleratesNotPublished:
    """``NeuronpediaClient.get_descriptions`` dispatches each feature
    through ``fetch_feature_description`` in a ``ThreadPoolExecutor`` and
    catches per-feature failures so one bad feature doesn't sink the
    batch (a dropped feature is simply absent from the result dict).

    ``NeuronpediaNotPublishedError`` is the EXPECTED state for a
    gemma-scope-2 feature Neuronpedia hasn't ingested (a 404). It must be
    tolerated exactly like the ``NeuronpediaError`` family: skipped,
    counted as dropped, and omitted from the result — never escaping as
    an uncaught crash that kills the whole batch eval.
    """

    def test_one_unpublished_feature_does_not_crash_batch(
        self, tmp_path,
    ) -> None:
        published = {1: "desc-1", 3: "desc-3"}

        def fake_fetch(model_id, sae_source, fid, *, config=None):
            if fid == 2:
                raise NeuronpediaNotPublishedError(
                    f"Neuronpedia does not host feature {fid}"
                )
            return published[fid]

        with patch.dict("os.environ", {"NEURONPEDIA_API_KEY": "np-test"}):
            with patch(
                "steering.eval.neuronpedia_client.fetch_feature_description",
                side_effect=fake_fetch,
            ):
                client = NeuronpediaClient(
                    "gemma-3-4b-it",
                    "22-gemmascope-2-res-16k",
                    tmp_path / "cache.yaml",
                    config=NeuronpediaConfig(max_retries=0),
                )
                descs = client.get_descriptions([1, 2, 3])

        # Batch completed; the un-ingested feature is skipped (absent),
        # the published ones are returned.
        assert descs == {1: "desc-1", 3: "desc-3"}
        assert 2 not in descs

    def test_real_failures_still_dropped_not_swallowed_silently(
        self, tmp_path,
    ) -> None:
        """A genuine ``NeuronpediaFetchError`` (not the expected-404
        case) must keep its existing skip-and-count behavior — the fix
        must not over-broaden or reclassify real fetch failures."""

        def fake_fetch(model_id, sae_source, fid, *, config=None):
            if fid == 2:
                raise NeuronpediaFetchError("malformed JSON")
            return f"desc-{fid}"

        with patch.dict("os.environ", {"NEURONPEDIA_API_KEY": "np-test"}):
            with patch(
                "steering.eval.neuronpedia_client.fetch_feature_description",
                side_effect=fake_fetch,
            ):
                client = NeuronpediaClient(
                    "gemma-3-4b-it",
                    "22-gemmascope-2-res-16k",
                    tmp_path / "cache.yaml",
                    config=NeuronpediaConfig(max_retries=0),
                )
                descs = client.get_descriptions([1, 2, 3])

        assert descs == {1: "desc-1", 3: "desc-3"}
        assert 2 not in descs
