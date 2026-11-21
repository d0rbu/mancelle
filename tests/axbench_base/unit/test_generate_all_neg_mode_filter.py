"""Unit tests for the ``run_negative_mode`` filter in ``_filter_methods_for_targets``.

Root-cause test suite for the Stage 4 bug where ``_filter_methods_for_targets``
did not check ``cfg.run_negative_mode``, causing ``generate_all`` to iterate
negative-mode methods even when negative-mode artifacts (concept_features_neg,
negative baselines, synth_training_negative) were never produced by Stages
1c/2/3.  The downstream crash was a ``FileNotFoundError`` or
``AxBenchBaseError`` when ``run_intervention`` tried to load the missing
negative-mode direction or adapter file.

The fix adds a ``run_negative_mode: bool`` parameter to
``_filter_methods_for_targets`` and drops every method whose
``mode == SteeringMode.NEGATIVE`` when the flag is ``False``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from steering.eval.axbench_base import generation as generation_mod
from steering.eval.axbench_base.generation import (
    _filter_methods_for_targets,
)
from steering.eval.axbench_base.generation import (
    generate_all as _generate_all,
)
from steering.eval.axbench_base.generation_helpers import GenerationResult
from steering.eval.axbench_base.method_registry import ALL_METHODS
from steering.eval.axbench_base.types import (
    Concept,
    SeedRow,
    SteeringMode,
    TargetBundle,
    TargetModelMetadata,
)
from tests.axbench_base import paths as test_paths
from tests.axbench_base.unit.test_config import make_valid_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_all(*args: Any, apply_max_act_scale: bool = False, **kwargs: Any) -> Path:
    return _generate_all(
        *args,
        apply_max_act_scale=apply_max_act_scale,
        **kwargs,
    )


def _make_stub_metas() -> dict[str, TargetModelMetadata]:
    """Build a minimal target_model_metas dict for filter tests.

    decoder_only: num_buckets=None, negative_selection=None
    encdec: num_buckets=4, negative_selection=None
    """
    return {
        "decoder_only": TargetModelMetadata(
            sae_layer=8,
            segment_size=128,
            num_buckets=None,
            n_sae_features=16384,
            negative_selection=None,
            feature_selection=None,
            per_feature_stats=None,
            device="cpu",
        ),
        "encdec": TargetModelMetadata(
            sae_layer=8,
            segment_size=128,
            num_buckets=4,
            n_sae_features=16384,
            negative_selection=None,
            feature_selection=None,
            per_feature_stats=None,
            device="cpu",
        ),
    }


def _make_stub_metas_with_neg_sampling() -> dict[str, TargetModelMetadata]:
    """Same as ``_make_stub_metas`` but with ``negative_selection`` set so
    the ``requires_negative_sampling`` filter passes for all methods.
    """
    metas = _make_stub_metas()
    metas["encdec"] = dataclasses.replace(
        metas["encdec"],
        negative_selection={"distribution": "geometric", "mean": 8},
    )
    return metas


def _concept(concept_id: int = 0) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=concept_id * 11,
        layer=8,
        sae_source="stub",
        description="the letter e",
        corpus_count=100,
        description_source="stub",
    )


def _seed_row() -> SeedRow:
    return SeedRow(
        shard_idx=0,
        doc_idx=0,
        source_nanochat_token_count=4,
        prefix_source_token_start=0,
        prefix_source_token_end=3,
        prefix_text="prefix",
        prefix_content_nanochat_tokens=[1, 2, 3],
        sae_feature_ids=[],
        sae_feature_mags=[],
        continuation_text="cont",
        continuation_content_nanochat_tokens=[7],
    )


@pytest.fixture(autouse=True)
def _strict_stage4_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        generation_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    monkeypatch.setattr(
        generation_mod,
        "_validate_stage4_context_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        generation_mod,
        "load_prompting_descriptions",
        lambda _cfg, _mode: {0: "description"},
    )


class _Stage4Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(sequence_len=2048)


def _stage4_model() -> _Stage4Model:
    return _Stage4Model()


class _ContextProvider:
    def get(self, token_ids: Sequence[int]) -> tuple[()]:
        del token_ids
        return ()


def _stage4_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.get_bos_token_id.return_value = 1
    tokenizer.encode.side_effect = lambda text: [10 + index for index, _ in enumerate(text)]
    tokenizer.decode.side_effect = lambda tokens: "x" * len(tokens)
    return tokenizer

# ---------------------------------------------------------------------------
# Unit tests: _filter_methods_for_targets with run_negative_mode
# ---------------------------------------------------------------------------


class TestFilterMethodsNegModeFlag:
    """Tests for the ``run_negative_mode`` parameter of
    ``_filter_methods_for_targets``.
    """

    def test_filter_methods_excludes_negative_when_run_negative_mode_false(
        self,
    ) -> None:
        """When run_negative_mode=False, _filter_methods_for_targets must
        exclude every method whose mode is SteeringMode.NEGATIVE.

        This is the root-cause test for the pipeline bug where Stage 4
        crashed on FileNotFoundError because it tried to load negative-mode
        artifacts that Stage 1c/2/3 never produced.
        """
        methods = _filter_methods_for_targets(
            ALL_METHODS,
            enabled_methods=None,
            disabled_methods=[],
            target_model_metas=_make_stub_metas(),
            run_negative_mode=False,
        )

        negative_methods = [
            m for m in methods if m.mode == SteeringMode.NEGATIVE
        ]
        assert len(negative_methods) == 0, (
            f"Expected 0 negative methods when run_negative_mode=False, "
            f"got {len(negative_methods)}: {[m.name for m in negative_methods]}"
        )

    def test_filter_methods_includes_negative_when_run_negative_mode_true(
        self,
    ) -> None:
        """When run_negative_mode=True, negative methods must be included
        (assuming no other filter excludes them).
        """
        methods = _filter_methods_for_targets(
            ALL_METHODS,
            enabled_methods=None,
            disabled_methods=[],
            target_model_metas=_make_stub_metas(),
            run_negative_mode=True,
        )

        negative_methods = [
            m for m in methods if m.mode == SteeringMode.NEGATIVE
        ]
        assert len(negative_methods) > 0, (
            "Expected at least one negative method when run_negative_mode=True"
        )

    def test_filter_methods_positive_count_unchanged_by_neg_mode_flag(
        self,
    ) -> None:
        """The positive-method count must be identical regardless of
        run_negative_mode -- the flag only affects negative methods.
        """
        metas = _make_stub_metas()

        pos_with_neg_on = [
            m
            for m in _filter_methods_for_targets(
                ALL_METHODS,
                enabled_methods=None,
                disabled_methods=[],
                target_model_metas=metas,
                run_negative_mode=True,
            )
            if m.mode == SteeringMode.POSITIVE
        ]
        pos_with_neg_off = [
            m
            for m in _filter_methods_for_targets(
                ALL_METHODS,
                enabled_methods=None,
                disabled_methods=[],
                target_model_metas=metas,
                run_negative_mode=False,
            )
            if m.mode == SteeringMode.POSITIVE
        ]
        assert len(pos_with_neg_on) == len(pos_with_neg_off), (
            f"Positive count should be identical: "
            f"neg_on={len(pos_with_neg_on)}, neg_off={len(pos_with_neg_off)}"
        )

    def test_filter_methods_neg_mode_off_with_enabled_methods_containing_negative(
        self,
    ) -> None:
        """If run_negative_mode=False but enabled_methods explicitly names a
        negative method, the negative method should STILL be excluded.

        The mode flag takes precedence over the enabled list -- if negative
        mode is off, no negative artifacts exist regardless of what the
        operator requested.
        """
        methods = _filter_methods_for_targets(
            ALL_METHODS,
            enabled_methods=[
                "neg_diffmean_decoder_only",
                "diffmean_decoder_only",
            ],
            disabled_methods=[],
            target_model_metas=_make_stub_metas(),
            run_negative_mode=False,
        )
        assert all(m.mode == SteeringMode.POSITIVE for m in methods), (
            "All surviving methods must be positive-mode when "
            "run_negative_mode=False"
        )
        assert any(m.name == "diffmean_decoder_only" for m in methods), (
            "The positive method 'diffmean_decoder_only' should survive"
        )

    def test_filter_methods_neg_mode_off_total_count_is_positive_only(
        self,
    ) -> None:
        """With neg mode off and no other filters, the method count should
        equal the number of positive methods in ALL_METHODS.

        Uses metas with neg_sampling so the requires_negative_sampling
        filter does not further reduce the count.
        """
        expected_positive = sum(
            1 for m in ALL_METHODS if m.mode == SteeringMode.POSITIVE
        )
        methods = _filter_methods_for_targets(
            ALL_METHODS,
            enabled_methods=None,
            disabled_methods=[],
            target_model_metas=_make_stub_metas_with_neg_sampling(),
            run_negative_mode=False,
        )
        assert len(methods) == expected_positive, (
            f"Expected {expected_positive} positive methods, "
            f"got {len(methods)}"
        )

    def test_filter_methods_neg_mode_on_total_count_is_all(self) -> None:
        """With neg mode on, all filters passing, and all targets having
        negative sampling, every method in ALL_METHODS survives the filter.
        """
        methods = _filter_methods_for_targets(
            ALL_METHODS,
            enabled_methods=None,
            disabled_methods=[],
            target_model_metas=_make_stub_metas_with_neg_sampling(),
            run_negative_mode=True,
        )
        assert len(methods) == len(ALL_METHODS), (
            f"Expected {len(ALL_METHODS)} methods with all filters passing, "
            f"got {len(methods)}"
        )


# ---------------------------------------------------------------------------
# Integration-level tests: generate_all respects run_negative_mode
# ---------------------------------------------------------------------------


class TestGenerateAllNegModeIntegration:
    """End-to-end tests verifying that ``generate_all`` correctly propagates
    ``run_negative_mode=False`` to the filter, so no negative-mode artifacts
    are ever touched.
    """

    def test_generate_all_with_neg_mode_off_does_not_touch_negative_artifacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """generate_all with run_negative_mode=False must NOT attempt to
        load any negative-mode artifact (no concept_features_neg, no
        negative baselines, no negative concept features).

        We patch run_intervention to record the methods it is called with,
        then assert no method.mode == 'negative' was ever dispatched.
        """
        cfg = make_valid_config(
            run_tag="rt",
            output_root=str(tmp_path / "eval_out"),
            run_negative_mode=False,
        )

        dispatched_methods: list[Any] = []

        def _fake_intervention(**kwargs: Any) -> GenerationResult:
            dispatched_methods.append(kwargs["method"])
            return GenerationResult(
                text="generated", token_count=4, stop_reason="bos_token",
            )

        monkeypatch.setattr(
            "steering.eval.axbench_base.generation.run_intervention",
            _fake_intervention,
        )

        model = _stage4_model()
        tokenizer = _stage4_tokenizer()

        targets = [
            TargetBundle(
                name="decoder_only",
                model=model,
                meta=_make_stub_metas_with_neg_sampling()["decoder_only"],

                tokenizer=tokenizer,
                context_provider=None,
            ),
            TargetBundle(
                name="encdec",
                model=model,
                meta=_make_stub_metas_with_neg_sampling()["encdec"],

                tokenizer=tokenizer,
                context_provider=_ContextProvider(),
            ),
        ]

        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=targets,
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[_seed_row()],
        )

        neg_dispatched = [
            m for m in dispatched_methods
            if m.mode == SteeringMode.NEGATIVE
        ]
        assert len(neg_dispatched) == 0, (
            f"Expected 0 negative methods dispatched, got "
            f"{len(neg_dispatched)}: {[m.name for m in neg_dispatched]}"
        )

    def test_generate_all_with_neg_mode_off_succeeds_without_negative_artifacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression test for the original bug: running generate_all with
        run_negative_mode=False should succeed even when negative-mode
        artifacts DO NOT EXIST on disk (because they were never produced
        by Stages 1c/2/3). Before the fix, this raised FileNotFoundError.
        """
        cfg = make_valid_config(
            run_tag="rt",
            output_root=str(tmp_path / "eval_out"),
            run_negative_mode=False,
        )

        def _fake_intervention(**kwargs: Any) -> GenerationResult:
            return GenerationResult(
                text="ok", token_count=2, stop_reason="bos_token",
            )

        monkeypatch.setattr(
            "steering.eval.axbench_base.generation.run_intervention",
            _fake_intervention,
        )

        model = _stage4_model()
        tokenizer = _stage4_tokenizer()

        targets = [
            TargetBundle(
                name="decoder_only",
                model=model,
                meta=_make_stub_metas()["decoder_only"],

                tokenizer=tokenizer,
                context_provider=None,
            ),
            TargetBundle(
                name="encdec",
                model=model,
                meta=_make_stub_metas()["encdec"],

                tokenizer=tokenizer,
                context_provider=_ContextProvider(),
            ),
        ]

        # Should NOT raise, even though no negative-mode artifacts exist.
        result_dir = generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=targets,
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[],
        )
        assert result_dir.exists(), (
            f"Expected generations dir to exist at {result_dir}"
        )

    def test_concept_features_negative_path_not_checked_when_neg_mode_off(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The concept_features_paths dict eagerly resolves both modes, but
        with neg mode off, the negative path is never USED. Verify by NOT
        creating concept_features_neg.safetensors and asserting no error.
        """
        cfg = make_valid_config(
            run_tag="rt",
            output_root=str(tmp_path / "eval_out"),
            run_negative_mode=False,
        )

        accessed_paths: list[Path] = []

        def _fake_intervention(**kwargs: Any) -> GenerationResult:
            accessed_paths.append(kwargs["concept_features_path"])
            return GenerationResult(
                text="ok", token_count=2, stop_reason="bos_token",
            )

        monkeypatch.setattr(
            "steering.eval.axbench_base.generation.run_intervention",
            _fake_intervention,
        )

        model = _stage4_model()
        tokenizer = _stage4_tokenizer()

        targets = [
            TargetBundle(
                name="decoder_only",
                model=model,
                meta=_make_stub_metas()["decoder_only"],

                tokenizer=tokenizer,
                context_provider=None,
            ),
            TargetBundle(
                name="encdec",
                model=model,
                meta=_make_stub_metas()["encdec"],

                tokenizer=tokenizer,
                context_provider=_ContextProvider(),
            ),
        ]

        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=targets,
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[],
        )

        # All accessed paths must be the positive concept features path,
        # never the negative one.
        neg_path = test_paths.default_paths("rt").concept_features_neg()
        for p in accessed_paths:
            assert p != neg_path, (
                f"concept_features_neg path was passed to run_intervention "
                f"even though run_negative_mode=False: {p}"
            )
