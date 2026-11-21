"""Tests for completion steering judge and dose-response analysis."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from errors import ParseError
from steering.eval.completion_judge import (
    CompletionJudgeClient,
    JudgementScore,
    build_judge_prompt,
    compute_dose_response,
    parse_judge_response,
    plot_ablation_comparison,
    plot_dose_response,
)


class TestBuildJudgePrompt:
    def test_includes_all_fields(self):
        system, user = build_judge_prompt(
            context="The cat sat on the mat.",
            completion="Then the dog arrived.",
            feature_description="references to animals",
            ground_truth="Then the dog came in.",
        )
        assert "cat sat on the mat" in user
        assert "dog arrived" in user
        assert "references to animals" in user
        assert "dog came in" in user
        assert "coherence" in system.lower()

    def test_returns_system_and_user_separately(self):
        system, user = build_judge_prompt(
            context="ctx",
            completion="comp",
            feature_description="desc",
            ground_truth="gt",
        )
        assert isinstance(system, str)
        assert isinstance(user, str)
        # System prompt should have rating instructions, user should have content
        assert "coherence" in system.lower()
        assert "ctx" in user

    def test_truncates_long_context(self):
        long_ctx = "word " * 5000  # 25000 chars, exceeds _MAX_CONTEXT_CHARS
        system, user = build_judge_prompt(
            context=long_ctx,
            completion="short",
            feature_description="test",
            ground_truth="truth",
        )
        assert len(user) < len(long_ctx)

    def test_excludes_ground_truth(self):
        system, user = build_judge_prompt(
            context="The cat sat.",
            completion="Then the dog.",
            feature_description="animals",
            ground_truth="This should not appear.",
            include_ground_truth=False,
        )
        assert "This should not appear" not in user
        assert "Ground Truth" not in user
        assert "ground truth" not in system.lower()
        assert "cat sat" in user
        assert "coherence" in system.lower()


class TestParseJudgeResponse:
    def test_valid_scores(self):
        response = "coherence: 4\nfeature_alignment: 3"
        score = parse_judge_response(response)
        assert score.coherence == 4
        assert score.feature_alignment == 3

    def test_valid_scores_max(self):
        response = "coherence: 5\nfeature_alignment: 5"
        score = parse_judge_response(response)
        assert score.coherence == 5
        assert score.feature_alignment == 5

    def test_out_of_range_clamps(self):
        response = "coherence: 7\nfeature_alignment: 0"
        score = parse_judge_response(response)
        assert score.coherence == 5
        assert score.feature_alignment == 1

    def test_missing_field_raises(self):
        response = "coherence: 4"
        with pytest.raises(ParseError, match="missing score pattern"):
            parse_judge_response(response)

    def test_garbage_raises(self):
        with pytest.raises(ParseError, match="missing score pattern"):
            parse_judge_response("I refuse to rate this")


class TestDoseResponse:
    def test_computes_mean_per_bucket(self):
        scores = {
            0: [JudgementScore(4, 2), JudgementScore(4, 4)],
            1: [JudgementScore(3, 3), JudgementScore(3, 5)],
        }
        result = compute_dose_response(scores)
        assert result["bucket_means"][0] == pytest.approx(3.0)
        assert result["bucket_means"][1] == pytest.approx(4.0)

    def test_empty_bucket_is_nan(self):
        import math

        scores = {0: [JudgementScore(4, 3)], 1: []}
        result = compute_dose_response(scores)
        assert result["bucket_means"][0] == pytest.approx(3.0)
        assert math.isnan(result["bucket_means"][1])


class TestPlotDoseResponse:
    def test_creates_plot_file(self, tmp_path):
        dose_data = {
            "bucket_means": [2.0, 3.0, 4.0],
            "bucket_stderrs": [0.2, 0.3, 0.1],
            "coherence_means": [4.0, 4.0, 4.0],
        }
        out = tmp_path / "plot.png"
        plot_dose_response(
            dose_data,
            feature_id=42,
            feature_description="test feature",
            output_path=out,
        )
        assert out.is_file()
        assert out.stat().st_size > 100

    def test_with_real_baseline(self, tmp_path):
        dose_data = {
            "bucket_means": [2.0, 3.0, 4.0],
            "bucket_stderrs": [0.2, 0.3, 0.1],
            "coherence_means": [4.0, 4.0, 4.0],
        }
        out = tmp_path / "plot_with_real.png"
        plot_dose_response(
            dose_data,
            feature_id=42,
            feature_description="test",
            output_path=out,
            real_baseline=3.5,
        )
        assert out.is_file()

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "subdir" / "nested" / "plot.png"
        dose_data = {
            "bucket_means": [2.0, 3.0],
            "bucket_stderrs": [0.2, 0.3],
            "coherence_means": [4.0, 4.0],
        }
        plot_dose_response(
            dose_data,
            feature_id=1,
            feature_description="test",
            output_path=out,
        )
        assert out.is_file()


class TestPlotAblationComparison:
    def test_creates_plot_file(self, tmp_path):
        out = tmp_path / "ablation.png"
        plot_ablation_comparison(
            real_mean=4.0,
            ablated_mean=2.5,
            feature_id=42,
            feature_description="test feature",
            output_path=out,
        )
        assert out.is_file()
        assert out.stat().st_size > 100

    def test_with_error_bars(self, tmp_path):
        out = tmp_path / "ablation_err.png"
        plot_ablation_comparison(
            real_mean=4.0,
            ablated_mean=2.5,
            feature_id=42,
            feature_description="test",
            output_path=out,
            real_stderr=0.2,
            ablated_stderr=0.3,
        )
        assert out.is_file()


class TestCompletionJudgeClient:
    def test_init_creates_cache_db(self, tmp_path):
        cache = str(tmp_path / "test_judge.db")
        CompletionJudgeClient(cache_path=cache)
        assert Path(cache).exists()

    def test_judge_uses_cache(self, tmp_path):
        """Second call with same inputs should return from cache without API."""
        cache = str(tmp_path / "test_judge.db")
        client = CompletionJudgeClient(cache_path=cache, requests_per_minute=0)

        # Pre-populate cache with a valid response
        system, user = build_judge_prompt(
            "context", "completion", "feature desc", "ground truth"
        )
        cache_payload = f"{system}\n---\n{user}"
        key = client._cache_key(cache_payload)
        client._cache_put(key, "coherence: 4\nfeature_alignment: 3")

        result = client.judge("context", "completion", "feature desc", "ground truth")
        assert result is not None
        assert result.coherence == 4
        assert result.feature_alignment == 3

    def test_judge_calls_api_on_miss(self, tmp_path):
        """On cache miss, should call the API and parse the response."""
        cache = str(tmp_path / "test_judge.db")
        client = CompletionJudgeClient(cache_path=cache, requests_per_minute=0)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "coherence: 5\nfeature_alignment: 4"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "steering.eval.completion_judge._requests.post", return_value=mock_response
        ):
            result = client.judge("ctx", "comp", "desc", "truth")

        assert result is not None
        assert result.coherence == 5
        assert result.feature_alignment == 4

    def test_judge_raises_on_exhausted_retries(self, tmp_path):
        """Should raise ParseError after 3 unparseable responses, not return None."""
        cache = str(tmp_path / "test_judge.db")
        client = CompletionJudgeClient(cache_path=cache, requests_per_minute=0)

        bad_resp = MagicMock()
        bad_resp.json.return_value = {"choices": [{"message": {"content": "garbage"}}]}
        bad_resp.raise_for_status = MagicMock()

        with patch(
            "steering.eval.completion_judge._requests.post", return_value=bad_resp
        ) as mock_post:
            with pytest.raises(ParseError, match="after 3 attempts"):
                client.judge("ctx", "comp", "desc", "truth")

        assert mock_post.call_count == 3

    def test_corrupt_cache_logs_and_refetches(self, tmp_path):
        """Corrupt cached response should log a warning and re-fetch from API."""
        cache = str(tmp_path / "test_judge.db")
        client = CompletionJudgeClient(cache_path=cache, requests_per_minute=0)

        # Pre-populate cache with a corrupt response
        system, user = build_judge_prompt("ctx", "comp", "desc", "truth")
        cache_payload = f"{system}\n---\n{user}"
        key = client._cache_key(cache_payload)
        client._cache_put(key, "THIS IS CORRUPT GARBAGE")

        # Set up API to return a valid response
        good_resp = MagicMock()
        good_resp.json.return_value = {
            "choices": [{"message": {"content": "coherence: 3\nfeature_alignment: 4"}}]
        }
        good_resp.raise_for_status = MagicMock()

        with patch(
            "steering.eval.completion_judge._requests.post", return_value=good_resp
        ) as mock_post:
            result = client.judge("ctx", "comp", "desc", "truth")

        # Should have called the API since cache was corrupt
        assert mock_post.call_count == 1
        assert result is not None
        assert result.coherence == 3
        assert result.feature_alignment == 4

    def test_corrupt_cache_overwrites_with_good_response(self, tmp_path):
        """After re-fetching due to corrupt cache, the new valid response
        should overwrite the corrupt cache entry."""
        cache = str(tmp_path / "test_judge.db")
        client = CompletionJudgeClient(cache_path=cache, requests_per_minute=0)

        # Pre-populate cache with corrupt data
        system, user = build_judge_prompt("ctx", "comp", "desc", "truth")
        cache_payload = f"{system}\n---\n{user}"
        key = client._cache_key(cache_payload)
        client._cache_put(key, "CORRUPT")

        good_resp = MagicMock()
        good_resp.json.return_value = {
            "choices": [{"message": {"content": "coherence: 5\nfeature_alignment: 5"}}]
        }
        good_resp.raise_for_status = MagicMock()

        with patch(
            "steering.eval.completion_judge._requests.post", return_value=good_resp
        ):
            client.judge("ctx", "comp", "desc", "truth")

        # Second call should use the now-valid cache (no API call)
        with patch(
            "steering.eval.completion_judge._requests.post"
        ) as mock_post:
            result = client.judge("ctx", "comp", "desc", "truth")

        mock_post.assert_not_called()
        assert result is not None
        assert result.coherence == 5

    def test_cache_key_deterministic(self, tmp_path):
        """Same prompt should produce same cache key."""
        cache = str(tmp_path / "test_judge.db")
        client = CompletionJudgeClient(cache_path=cache)
        k1 = client._cache_key("hello world")
        k2 = client._cache_key("hello world")
        k3 = client._cache_key("different")
        assert k1 == k2
        assert k1 != k3
