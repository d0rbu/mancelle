"""Slice 14 audit X2: functional tests for ``steering/eval/sanity_check/run_all.py``.

The real ``run_all.main()`` pipeline loads a multi-hundred-MB checkpoint,
runs DCLM/ChatCORE/harmfulness, and calls an HTTPS judge. None
of that is reproducible in unit tests, but the *orchestration* in
``main()`` — argparse parsing, source loop, dispatch toggles, results
aggregation, YAML serialisation — is pure dispatch logic over the
helper functions and IS testable.

These tests pin that dispatch logic by:

1. Monkey-patching every heavy callee (``load_decoder_model``,
   ``test_inference``, ``test_dclm``, ``test_chatcore``,
   ``test_harmfulness``, ``InferenceEngine``,
   ``load_project_dotenv``, ``torch.cuda.empty_cache``,
   ``autodetect_device_type``) to deterministic stubs that record their
   invocations.
2. Driving the argparse-flavoured ``main()`` via ``sys.argv``
   monkey-patching.
3. Asserting on which stubs were called for each flag combination,
   the per-source result envelope, and the final ``results.yaml``
   contents.

A separate :class:`TestStaticContract` block keeps the cheap import /
signature / MODELS-table guards from the earlier shape-only suite,
because losing them silently is a regression in its own right.
"""

from __future__ import annotations

import inspect
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
import yaml

from steering.eval.harmfulness_judge import ComplianceJudgeResult, JudgeResult
from steering.eval.sanity_check import run_all

# ---------------------------------------------------------------------------
# Static contract (cheap, no monkeypatching)
# ---------------------------------------------------------------------------


class TestStaticContract:
    """Cheap shape-only guards kept from the prior reduced-scope suite."""

    def test_module_imports_with_no_side_effects(self) -> None:
        """No network / FS reads / global mutation at import time."""
        assert callable(run_all.main)

    def test_models_table_documented(self) -> None:
        assert set(run_all.MODELS) == {"base", "sft"}
        for info in run_all.MODELS.values():
            assert "expected_core" in info
            assert "label" in info
            if info["expected_core"] is not None:
                assert isinstance(info["expected_core"], (int, float))
            assert isinstance(info["label"], str)

    @pytest.mark.parametrize(
        "name",
        [
            "test_inference",
            "test_dclm",
            "test_harmfulness",
            "test_chatcore",
            "load_decoder_model",
            "_resolve_ckpt_dir",
            "main",
        ],
    )
    def test_exposes_function(self, name: str) -> None:
        fn = getattr(run_all, name, None)
        assert callable(fn), f"Missing callable {name!r} on run_all"

    def test_main_has_no_positional_args(self) -> None:
        """``main()`` is argparse-driven; pins zero parameters."""
        sig = inspect.signature(run_all.main)
        assert sig.parameters == {}, (
            f"Expected zero-parameter argparse-driven main; got {sig.parameters}"
        )

    def test_resolve_ckpt_dir_rejects_unknown_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_all, "get_base_dir", lambda: str(tmp_path))
        monkeypatch.setattr(run_all, "find_largest_model", lambda *_a, **_kw: "d12")
        with pytest.raises(ValueError, match="Unknown source"):
            run_all._resolve_ckpt_dir("not-a-real-source")


# ---------------------------------------------------------------------------
# Functional dispatch tests (full main() invocation with stubbed callees)
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Patch every heavy callee so ``main()`` runs purely on its dispatch logic.

    Returns a ``_StubCallRecorder`` whose attributes count and record the
    arguments of each helper invocation, plus a ``read_yaml()`` accessor
    that loads the final ``results.yaml`` from the tmp working directory.
    """
    # Each source's stubbed model is a unique sentinel so the test can
    # tell base vs sft loads apart.
    fake_models: dict[str, tuple[object, object, dict, int]] = {
        "base": (object(), object(), {"step": 100}, 100),
        "sft": (object(), object(), {"step": 200}, 200),
    }

    class _StubCallRecorder:
        def __init__(self) -> None:
            self.load_decoder_calls: list[str] = []
            self.test_inference_calls: list[str] = []
            self.test_dclm_calls: list[str] = []
            self.test_chatcore_calls: list[str] = []
            self.test_harmfulness_calls: int = 0
            self.engine_constructed: int = 0
            self.empty_cache_calls: int = 0
            self.compute_init_calls: list[str] = []
            self.compute_cleanup_calls: int = 0

        def read_yaml(self) -> dict[str, Any]:
            path = tmp_path / "steering" / "eval" / "sanity_check" / "results.yaml"
            assert path.exists(), f"Expected results.yaml at {path}; not written"
            with open(path) as f:
                return yaml.safe_load(f)

    recorder = _StubCallRecorder()

    # Stub heavy callees. Each stub returns the documented result-dict
    # shape so main()'s post-processing (print summary) still works.

    def fake_load_decoder_model(source: str, _device):
        recorder.load_decoder_calls.append(source)
        return fake_models[source]

    def fake_test_inference(_engine, _ctx, source, *_a, **_kw):
        recorder.test_inference_calls.append(source)
        return {"status": "pass", "issues": [], "_source": source}

    def fake_test_dclm(_model, _tok, _device_type, _ctx, expected_core):
        recorder.test_dclm_calls.append("base" if expected_core is not None else "sft")
        return {
            "status": "pass",
            "core_score": 0.2461 if expected_core is not None else 0.0,
            "issues": [],
        }

    def fake_test_chatcore(_model, _tok, _ctx, max_problems: int = 50):
        recorder.test_chatcore_calls.append(f"max_problems={max_problems}")
        return {
            "status": "pass",
            "chatcore": 0.1234,
            "per_task_scores": {},
            "issues": [],
        }

    def fake_test_harmfulness(_engine, _ctx):
        recorder.test_harmfulness_calls += 1
        return {
            "status": "pass",
            "harmful_prompts": {"mean_score": 0.5, "num_scored": 1},
            "benign_prompts": {"mean_score": 0.1, "num_scored": 1},
            "per_prompt_scores": [],
            "issues": [],
        }

    def fake_engine(*_a, **_kw):
        recorder.engine_constructed += 1
        return object()

    def fake_empty_cache():
        recorder.empty_cache_calls += 1

    def fake_compute_init(device_type: str):
        recorder.compute_init_calls.append(device_type)
        return False, 0, 0, 1, t.device(device_type)

    def fake_compute_cleanup() -> None:
        recorder.compute_cleanup_calls += 1

    monkeypatch.setattr(run_all, "load_decoder_model", fake_load_decoder_model)
    monkeypatch.setattr(run_all, "test_inference", fake_test_inference)
    monkeypatch.setattr(run_all, "test_dclm", fake_test_dclm)
    monkeypatch.setattr(run_all, "test_chatcore", fake_test_chatcore)
    monkeypatch.setattr(run_all, "test_harmfulness", fake_test_harmfulness)
    monkeypatch.setattr(run_all, "InferenceEngine", fake_engine)
    monkeypatch.setattr(run_all, "load_project_dotenv", lambda: None)
    monkeypatch.setattr(run_all, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(run_all, "compute_init", fake_compute_init)
    monkeypatch.setattr(run_all, "compute_cleanup", fake_compute_cleanup)
    # ``run_all.torch.cuda.empty_cache`` is called unconditionally; patch
    # it on the actual cuda module since main() reaches torch.cuda.empty_cache().
    monkeypatch.setattr(t.cuda, "empty_cache", fake_empty_cache)
    # Pin OPENROUTER_API_KEY so the non-chatcore-only branch doesn't sys.exit.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    # ``main()`` writes ``Path("steering/eval/sanity_check/results.yaml")``
    # which is interpreted relative to CWD. Pin CWD to tmp_path so the
    # write lands under it.
    (tmp_path / "steering" / "eval" / "sanity_check").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    return run_all, recorder


def _set_argv(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["run_all", *args])


class TestMainFullRun:
    """``main()`` with no flags runs every test on both sources."""

    def test_runs_all_4_tests_for_both_sources(
        self, stub_main, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_all, recorder = stub_main
        _set_argv(monkeypatch)

        run_all.main()

        # Both sources loaded in order.
        assert recorder.load_decoder_calls == ["base", "sft"]
        # All four inner tests ran for each source.
        assert recorder.test_inference_calls == ["base", "sft"]
        assert recorder.test_dclm_calls == ["base", "sft"]
        assert recorder.test_chatcore_calls == [
            "max_problems=50",
            "max_problems=50",
        ]
        assert recorder.test_harmfulness_calls == 2
        # An InferenceEngine is constructed per source.
        assert recorder.engine_constructed == 2
        # empty_cache called once per source to free between loads.
        assert recorder.empty_cache_calls == 2

        # YAML envelope contains both sources with the full test set.
        out = recorder.read_yaml()
        assert "sanity_check" in out
        results = out["sanity_check"]
        assert set(results) == {"base", "sft"}
        for source, source_result in results.items():
            assert "model_step" in source_result
            assert source_result["model_step"] == (100 if source == "base" else 200)
            assert {"inference", "dclm", "chatcore", "harmfulness"} <= set(source_result)

    def test_writes_results_yaml_with_top_level_key(
        self, stub_main, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_all, recorder = stub_main
        _set_argv(monkeypatch)

        run_all.main()
        out = recorder.read_yaml()
        # Documented top-level wrapper key.
        assert list(out) == ["sanity_check"]


class TestMainSourceFilter:
    """``--source=base`` / ``--source=sft`` restrict the loop to one model."""

    def test_only_base(self, stub_main, monkeypatch: pytest.MonkeyPatch) -> None:
        run_all, recorder = stub_main
        _set_argv(monkeypatch, "--source", "base")

        run_all.main()

        assert recorder.load_decoder_calls == ["base"]
        assert recorder.test_inference_calls == ["base"]
        # dclm called once with expected_core (base has 0.2461 expected).
        assert recorder.test_dclm_calls == ["base"]
        assert recorder.test_chatcore_calls == ["max_problems=50"]
        assert recorder.test_harmfulness_calls == 1
        out = recorder.read_yaml()["sanity_check"]
        assert set(out) == {"base"}

    def test_only_sft(self, stub_main, monkeypatch: pytest.MonkeyPatch) -> None:
        run_all, recorder = stub_main
        _set_argv(monkeypatch, "--source", "sft")

        run_all.main()

        assert recorder.load_decoder_calls == ["sft"]
        # SFT row in MODELS has expected_core=None, so the dclm stub
        # records "sft".
        assert recorder.test_dclm_calls == ["sft"]
        out = recorder.read_yaml()["sanity_check"]
        assert set(out) == {"sft"}


class TestMainSkipChatcore:
    """``--skip-chatcore`` drops the chatcore phase but keeps everything else."""

    def test_skip_chatcore_skips_only_chatcore(
        self, stub_main, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_all, recorder = stub_main
        _set_argv(monkeypatch, "--skip-chatcore")

        run_all.main()

        # Both sources still loaded.
        assert recorder.load_decoder_calls == ["base", "sft"]
        # ChatCORE was skipped — never called.
        assert recorder.test_chatcore_calls == []
        # Every other test ran twice (once per source).
        assert recorder.test_inference_calls == ["base", "sft"]
        assert recorder.test_dclm_calls == ["base", "sft"]
        assert recorder.test_harmfulness_calls == 2

        # YAML has no chatcore key per source.
        for source_result in recorder.read_yaml()["sanity_check"].values():
            assert "chatcore" not in source_result
            assert "inference" in source_result
            assert "dclm" in source_result
            assert "harmfulness" in source_result


class TestMainChatcoreOnly:
    """``--chatcore-only`` runs ONLY ChatCORE; no API key required."""

    def test_chatcore_only_short_circuits_other_tests(
        self,
        stub_main,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_all, recorder = stub_main
        # Specifically REMOVE the API key — chatcore-only must not require it.
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        _set_argv(monkeypatch, "--chatcore-only")

        run_all.main()

        # Both sources loaded, but only chatcore runs.
        assert recorder.load_decoder_calls == ["base", "sft"]
        assert recorder.test_chatcore_calls == [
            "max_problems=50",
            "max_problems=50",
        ]
        # All other phases skipped.
        assert recorder.test_inference_calls == []
        assert recorder.test_dclm_calls == []
        assert recorder.test_harmfulness_calls == 0
        # No InferenceEngine constructed in chatcore-only mode.
        assert recorder.engine_constructed == 0
        assert recorder.compute_init_calls == []
        assert recorder.compute_cleanup_calls == 0

        for source_result in recorder.read_yaml()["sanity_check"].values():
            assert "chatcore" in source_result
            assert "inference" not in source_result

    def test_nonzero_rank_does_not_write_results(
        self,
        stub_main,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_all, recorder = stub_main
        monkeypatch.setattr(
            run_all,
            "get_dist_info",
            lambda: (True, 1, 1, 2),
        )

        def fake_ddp_compute_init(device_type: str):
            recorder.compute_init_calls.append(device_type)
            return True, 1, 1, 2, t.device("cpu")

        monkeypatch.setattr(
            run_all,
            "compute_init",
            fake_ddp_compute_init,
        )
        _set_argv(monkeypatch, "--chatcore-only", "--source", "sft")

        run_all.main()

        assert recorder.test_chatcore_calls == ["max_problems=50"]
        result_path = Path.cwd() / "steering" / "eval" / "sanity_check" / "results.yaml"
        assert not result_path.exists()
        assert recorder.compute_init_calls == ["cpu"]
        assert recorder.compute_cleanup_calls == 1


class TestMainRequiresApiKey:
    """Non-chatcore-only branches require ``OPENROUTER_API_KEY``."""

    def test_exits_when_api_key_missing(
        self,
        stub_main,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _run_all, _recorder = stub_main
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        _set_argv(monkeypatch, "--source", "base")

        # main() prints + sys.exit(1) when needs_api_key and the env is
        # empty. SystemExit propagates as the dispatch failure signal.
        with pytest.raises(SystemExit) as exc_info:
            _run_all.main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "OPENROUTER_API_KEY" in captured.out

    def test_does_not_exit_when_api_key_missing_but_chatcore_only(
        self, stub_main, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Already covered by chatcore-only test, asserted explicitly
        here as a documentation pin: ``--chatcore-only`` does NOT
        require ``OPENROUTER_API_KEY``.
        """
        run_all, _recorder = stub_main
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        _set_argv(monkeypatch, "--chatcore-only")
        # Must not raise SystemExit.
        run_all.main()


class TestMainResultEnvelope:
    """The shape of the persisted YAML is part of the public contract."""

    def test_per_source_envelope_includes_model_step(
        self, stub_main, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_all, recorder = stub_main
        _set_argv(monkeypatch, "--source", "base")

        run_all.main()
        env = recorder.read_yaml()["sanity_check"]["base"]
        assert env["model_step"] == 100, (
            "main() must record the per-source step from load_decoder_model"
        )

    def test_inference_stub_payload_passes_through_to_yaml(
        self, stub_main, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each test_* return dict is written verbatim under its key."""
        run_all, recorder = stub_main
        _set_argv(monkeypatch, "--source", "base")

        run_all.main()
        env = recorder.read_yaml()["sanity_check"]["base"]
        assert env["inference"]["_source"] == "base"
        assert env["dclm"]["core_score"] == 0.2461
        assert env["chatcore"]["chatcore"] == 0.1234


class TestHarmfulnessComplianceSmoke:
    def test_scores_compliance_on_benign_pairs_only_with_luna(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        harmful = [{"prompt": "harm", "category": "Violence"}]
        benign = [
            {"prompt": "benign-a", "category": "benign"},
            {"prompt": "benign-b", "category": "benign"},
        ]
        monkeypatch.setattr(run_all, "load_harmful_prompts", lambda: harmful)

        def _load_benign(*, n: int, seed: int) -> list[dict[str, str]]:
            assert (n, seed) == (500, 42)
            return benign

        monkeypatch.setattr(run_all, "load_benign_prompts", _load_benign)

        class _Engine:
            @staticmethod
            def generate(prompt: str, **_kwargs: Any) -> str:
                return f"response-for-{prompt}"

        class _Judge:
            def __init__(self, *, model: str, requests_per_minute: int) -> None:
                assert model == "openai/gpt-5.6-luna"
                assert requests_per_minute == 60
                self.compliance_pairs: list[dict[str, str]] | None = None

            @staticmethod
            def score_all(pairs: list[dict[str, str]]) -> list[JudgeResult]:
                assert [pair["pool"] for pair in pairs] == [
                    "harmful",
                    "benign",
                    "benign",
                ]
                return [
                    JudgeResult([], 1, "Safe."),
                    JudgeResult([], 1, "Safe."),
                    JudgeResult([], 1, "Safe."),
                ]

            def score_compliance_all(
                self,
                pairs: list[dict[str, str]],
            ) -> list[ComplianceJudgeResult]:
                self.compliance_pairs = pairs
                assert [pair["prompt"] for pair in pairs] == [
                    "benign-a",
                    "benign-b",
                ]
                assert all(pair["pool"] == "benign" for pair in pairs)
                return [
                    ComplianceJudgeResult(5, False, "Complete."),
                    ComplianceJudgeResult(2, True, "Refusal."),
                ]

        judge = _Judge(model="openai/gpt-5.6-luna", requests_per_minute=60)

        def _judge_factory(**kwargs: Any) -> _Judge:
            assert kwargs == {
                "model": "openai/gpt-5.6-luna",
                "requests_per_minute": 60,
            }
            return judge

        monkeypatch.setattr(run_all, "JudgeClient", _judge_factory)

        result = run_all.test_harmfulness(
            cast(run_all.InferenceEngine, _Engine()),
            nullcontext(),
        )

        assert result["status"] == "pass"
        assert result["benign_prompts"]["compliance_rate"] == 0.5
        assert result["benign_prompts"]["compliance_num_scored"] == 2
        assert result["benign_prompts"]["compliance_num_failed"] == 0
        assert [
            record["is_compliant"] for record in result["benign_prompts"]["compliance_judgments"]
        ] == [True, False]
        assert result["issues"] == []

    def test_compliance_transport_failure_is_reported_separately(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            run_all,
            "load_harmful_prompts",
            lambda: [{"prompt": "harm", "category": "Violence"}],
        )
        monkeypatch.setattr(
            run_all,
            "load_benign_prompts",
            lambda *, n, seed: [
                {"prompt": "benign-a", "category": "benign"},
                {"prompt": "benign-b", "category": "benign"},
            ],
        )

        class _Engine:
            @staticmethod
            def generate(prompt: str, **_kwargs: Any) -> str:
                return f"response-for-{prompt}"

        class _Judge:
            @staticmethod
            def score_all(_pairs: list[dict[str, str]]) -> list[JudgeResult]:
                return [
                    JudgeResult([], 1, "Safe."),
                    JudgeResult([], 1, "Safe."),
                    JudgeResult([], 1, "Safe."),
                ]

            @staticmethod
            def score_compliance_all(
                _pairs: list[dict[str, str]],
            ) -> list[ComplianceJudgeResult | None]:
                return [ComplianceJudgeResult(5, False, "Complete."), None]

        monkeypatch.setattr(run_all, "JudgeClient", lambda **_kwargs: _Judge())
        result = run_all.test_harmfulness(
            cast(run_all.InferenceEngine, _Engine()),
            nullcontext(),
        )
        assert result["status"] == "pass"
        assert result["benign_prompts"]["compliance_num_scored"] == 1
        assert result["benign_prompts"]["compliance_num_failed"] == 1
        assert result["issues"] == ["1 benign compliance API calls failed"]


# ---------------------------------------------------------------------------
# Negative tests on the dispatch table — guard against silent regressions.
# ---------------------------------------------------------------------------


class TestMainSourceArgumentValidation:
    """``--source`` is constrained to ``{base, sft}`` by argparse."""

    def test_unknown_source_rejected_by_argparse(
        self, stub_main, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_all, _recorder = stub_main
        _set_argv(monkeypatch, "--source", "totally-fake")
        # argparse emits SystemExit(2) on invalid choice.
        with pytest.raises(SystemExit) as exc_info:
            run_all.main()
        assert exc_info.value.code == 2
