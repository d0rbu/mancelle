"""Unit tests for :func:`runtime_state.validate_env_for_stages`.

The helper is the Phase 5 gate that runs BEFORE any heavy model load so
operators see missing credentials immediately rather than after a
30-second Gemma download. The contract the CLI relies on:

- Stages that don't hit OpenRouter are a no-op regardless of env state.
- Stages that do hit OpenRouter raise :class:`AxBenchBaseError` when
  :data:`OPENROUTER_API_KEY` is missing, with an error message that
  names every triggering stage (not just the first one).
- Stage 0c is conditional: with examples QC disabled and
  ``verification_api_model=null``, it touches neither API. Enabling either
  examples QC or API verification generation requires both OpenRouter and
  Neuronpedia credentials.
- The helper never touches the real process environment — every test
  uses ``monkeypatch`` to set/delete the env var in an isolated scope.
"""

from __future__ import annotations

from typing import Any

import pytest

# NOTE: ``scripts.eval.axbench_base_eval`` is imported at module scope even
# though most tests reach it via ``cli.<name>``. The import triggers
# transitive loading of ``neuronpedia`` which auto-loads the project
# ``.env`` file the first time it runs. If we imported ``cli`` lazily
# inside a test function, ``monkeypatch.delenv`` would take effect and
# then the first test that reaches ``cli`` would clobber the env state
# by re-reading ``.env`` during import. Loading ``cli`` eagerly at
# module scope pins that side-effect BEFORE any monkeypatch runs, so
# the fixture-level ``delenv`` is reliable.
from scripts.eval.axbench_base_eval import _resolve_stage_list
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.openrouter import OPENROUTER_API_KEY_ENV
from steering.eval.axbench_base.runtime_state import (
    _STAGES_REQUIRING_OPENROUTER,
    validate_env_for_stages,
)
from steering.eval.axbench_base.types import AxBenchBaseError, Stage
from steering.eval.neuronpedia_client import NEURONPEDIA_API_KEY_ENV

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


def _make_cfg(**overrides: Any) -> AxBenchBaseConfig:
    """Return a minimal :class:`AxBenchBaseConfig` for validation tests.

    Every mandatory field must be present; optional fields are overridden
    on a per-test basis so the default is "QC enabled, API generation enabled,
    run_negative_mode true" which exercises the Stage 0c env path.
    """
    kwargs: dict[str, Any] = {
        "feature_dir": "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        "feature_statistics_path": "out/data/features/test/stats.safetensors",
        "decoder_only": {"source": "base", "model_tag": "d13", "step": 10000},
        "encdec": {"source": "encdec", "model_tag": "d13", "step": 10000},
        "device": "cpu",
        "output_root": "out/eval/axbench_base",
        "run_tag": "env-validation-test",
        "stage": "all",
        "run_negative_mode": True,
        "enable_concept_examples_qc": True,
    }
    kwargs.update(overrides)
    return AxBenchBaseConfig(**kwargs)


@pytest.fixture
def _env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure external API keys are NOT set for the duration of the test."""
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    monkeypatch.delenv(NEURONPEDIA_API_KEY_ENV, raising=False)


@pytest.fixture
def _env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure BOTH :data:`OPENROUTER_API_KEY` and
    :data:`NEURONPEDIA_API_KEY` are set for the duration of the test.
    The values are harmless placeholders; the helper only checks
    presence (via :func:`os.environ.get`), not format.

    Both are set together because Stage 0a needs Neuronpedia
    unconditionally and OpenRouter conditionally on QC; the previous
    one-key fixture passed locally only because the project ``.env``
    file is auto-loaded by the ``neuronpedia`` library import. In
    clean CI without ``.env``, that auto-load doesn't happen and tests
    that asserted "with env set, Stage 0a passes" would fail. Setting
    both eliminates that hidden dependency.

    Tests that specifically need only ONE key set use a dedicated
    fixture (see TestNeuronpediaEnvValidation below)."""
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-or-test-value")
    monkeypatch.setenv(NEURONPEDIA_API_KEY_ENV, "np-test-value")


# ---------------------------------------------------------------------------
# Happy paths — empty list / data-only stages / env set
# ---------------------------------------------------------------------------


class TestHappyPaths:
    def test_empty_stage_list_noop(self, _env_unset: None) -> None:
        """An empty list is a no-op regardless of env state."""
        validate_env_for_stages([])

    def test_empty_stage_list_noop_with_cfg(
        self, _env_unset: None,
    ) -> None:
        """Empty list + cfg still no-ops."""
        validate_env_for_stages([], cfg=_make_cfg())

    def test_prepare_seed_pools_alone_noop(
        self, _env_unset: None,
    ) -> None:
        """Stage 0b is pure data; no env requirement."""
        validate_env_for_stages([Stage.PREPARE_SEED_POOLS])

    def test_aggregate_alone_noop(self, _env_unset: None) -> None:
        """Stage 6 is pure disk IO — no env requirement."""
        validate_env_for_stages([Stage.AGGREGATE])

    def test_generate_alone_noop(self, _env_unset: None) -> None:
        """Stage 4 uses the target model InferenceEngine, not
        OpenRouter."""
        validate_env_for_stages([Stage.GENERATE])

    def test_stage_0a_with_openrouter_unset_noop_when_np_set(
        self, _env_unset: None, _np_env_set: None,
    ) -> None:
        """Stage 0a no longer touches OpenRouter; it only needs
        Neuronpedia for descriptions."""
        cfg = _make_cfg(enable_concept_examples_qc=True)
        validate_env_for_stages([Stage.PREPARE_CONCEPTS], cfg=cfg)

    def test_prepare_concepts_with_env_set_noop(
        self, _env_set: None,
    ) -> None:
        """Env is set -> Stage 0a description fetch preflight passes."""
        cfg = _make_cfg(enable_concept_examples_qc=True)
        validate_env_for_stages([Stage.PREPARE_CONCEPTS], cfg=cfg)

    def test_prepare_negative_prefixes_with_env_set_noop(
        self, _env_set: None,
    ) -> None:
        """Env is set → Stage 0c with examples QC on passes."""
        cfg = _make_cfg(enable_concept_examples_qc=True)
        validate_env_for_stages([Stage.PREPARE_NEGATIVE_PREFIXES], cfg=cfg)

    def test_prepare_negative_prefixes_default_api_mode_with_env_set_noop(
        self, _env_set: None,
    ) -> None:
        cfg = _make_cfg(
            enable_concept_examples_qc=False,
        )
        assert cfg.verification_api_model == "inclusionai/ling-2.6-flash"
        validate_env_for_stages([Stage.PREPARE_NEGATIVE_PREFIXES], cfg=cfg)

    def test_judge_with_env_set_noop(self, _env_set: None) -> None:
        """Env is set → Stage 5 passes."""
        validate_env_for_stages([Stage.JUDGE])

    def test_synth_positive_with_env_set_noop(
        self, _env_set: None,
    ) -> None:
        """Env is set → Stage 1b passes."""
        validate_env_for_stages([Stage.SYNTH_POSITIVE_MODE])

    def test_synth_negative_with_env_set_noop(
        self, _env_set: None,
    ) -> None:
        """Env is set → Stage 1c passes."""
        validate_env_for_stages([Stage.SYNTH_NEGATIVE_MODE])


# ---------------------------------------------------------------------------
# Raise paths — OpenRouter required but env missing
# ---------------------------------------------------------------------------


class TestRaisePaths:
    def test_prepare_negative_prefixes_without_env_raises(
        self, _env_unset: None,
    ) -> None:
        """Stage 0c + QC on + OpenRouter unset → raises."""
        cfg = _make_cfg(enable_concept_examples_qc=True)
        with pytest.raises(AxBenchBaseError, match=OPENROUTER_API_KEY_ENV):
            validate_env_for_stages(
                [Stage.PREPARE_NEGATIVE_PREFIXES], cfg=cfg,
            )

    def test_prepare_negative_prefixes_without_env_and_no_cfg_raises(
        self, _env_unset: None,
    ) -> None:
        """``cfg=None`` defensively treats Stage 0c as requiring env."""
        with pytest.raises(AxBenchBaseError, match=OPENROUTER_API_KEY_ENV):
            validate_env_for_stages([Stage.PREPARE_NEGATIVE_PREFIXES])

    def test_prepare_negative_prefixes_default_api_mode_requires_openrouter(
        self, _env_unset: None, _np_env_set: None,
    ) -> None:
        cfg = _make_cfg(
            enable_concept_examples_qc=False,
        )
        assert cfg.verification_api_model == "inclusionai/ling-2.6-flash"
        with pytest.raises(AxBenchBaseError, match=OPENROUTER_API_KEY_ENV):
            validate_env_for_stages(
                [Stage.PREPARE_NEGATIVE_PREFIXES],
                cfg=cfg,
            )

    def test_synth_positive_without_env_raises(
        self, _env_unset: None,
    ) -> None:
        """Stage 1b unconditionally raises without env."""
        with pytest.raises(AxBenchBaseError, match=OPENROUTER_API_KEY_ENV):
            validate_env_for_stages([Stage.SYNTH_POSITIVE_MODE])

    def test_synth_negative_without_env_raises(
        self, _env_unset: None,
    ) -> None:
        """Stage 1c unconditionally raises without env."""
        with pytest.raises(AxBenchBaseError, match=OPENROUTER_API_KEY_ENV):
            validate_env_for_stages([Stage.SYNTH_NEGATIVE_MODE])

    def test_judge_without_env_raises(self, _env_unset: None) -> None:
        """Stage 5 unconditionally raises without env."""
        with pytest.raises(AxBenchBaseError, match=OPENROUTER_API_KEY_ENV):
            validate_env_for_stages([Stage.JUDGE])

    def test_all_stages_without_env_raises(
        self, _env_unset: None,
    ) -> None:
        """An ``ALL``-expanded stage list (via ``_resolve_stage_list``)
        raises because Stage.JUDGE / 1b / 1c / 0c (QC on) all need env."""
        cfg = _make_cfg(enable_concept_examples_qc=True)
        stages = _resolve_stage_list(Stage.ALL, run_negative_mode=True)
        with pytest.raises(AxBenchBaseError, match=OPENROUTER_API_KEY_ENV):
            validate_env_for_stages(stages, cfg=cfg)

    def test_all_stages_without_env_also_raises_with_qc_off(
        self, _env_unset: None, _np_env_set: None,
    ) -> None:
        """Even with QC off, Stage 1b/1c/5 remain in the list so
        ``ALL`` still raises — but 0a is not a triggering stage."""
        cfg = _make_cfg(enable_concept_examples_qc=False)
        stages = _resolve_stage_list(Stage.ALL, run_negative_mode=True)
        with pytest.raises(AxBenchBaseError) as exc_info:
            validate_env_for_stages(stages, cfg=cfg)
        msg = str(exc_info.value)
        # 0a must not appear; default API-mode 0c must appear.
        assert "'0a'" not in msg
        assert "'0c'" in msg
        # 1b / 1c / 5 must appear.
        assert "'1b'" in msg
        assert "'1c'" in msg
        assert "'5'" in msg


# ---------------------------------------------------------------------------
# Mixed lists — zero/one/many OpenRouter stages
# ---------------------------------------------------------------------------


class TestMixedLists:
    def test_mixed_with_one_openrouter_stage_raises(
        self, _env_unset: None,
    ) -> None:
        """A list with at least one OpenRouter stage raises even if
        other stages in the list are pure data."""
        with pytest.raises(AxBenchBaseError, match=OPENROUTER_API_KEY_ENV):
            validate_env_for_stages(
                [Stage.AGGREGATE, Stage.JUDGE],
            )

    def test_mixed_with_zero_openrouter_stages_noop(
        self, _env_unset: None,
    ) -> None:
        """A list with no OpenRouter stages is a no-op."""
        validate_env_for_stages(
            [Stage.AGGREGATE, Stage.GENERATE],
        )

    def test_mixed_with_multiple_openrouter_stages_raises(
        self, _env_unset: None,
    ) -> None:
        """Two OpenRouter stages mixed with a data stage — the error
        must mention both triggering stages, not just the first."""
        with pytest.raises(AxBenchBaseError) as exc_info:
            validate_env_for_stages(
                [Stage.GENERATE, Stage.SYNTH_POSITIVE_MODE, Stage.JUDGE],
            )
        msg = str(exc_info.value)
        assert "'1b'" in msg
        assert "'5'" in msg


# ---------------------------------------------------------------------------
# Error message quality
# ---------------------------------------------------------------------------


class TestErrorMessageQuality:
    def test_error_message_names_missing_env_var(
        self, _env_unset: None,
    ) -> None:
        """The error message must literally contain ``OPENROUTER_API_KEY``
        so a grep over the operator's terminal points at the right fix."""
        with pytest.raises(AxBenchBaseError) as exc_info:
            validate_env_for_stages([Stage.JUDGE])
        assert "OPENROUTER_API_KEY" in str(exc_info.value)

    def test_error_message_lists_all_triggering_stages_not_just_first(
        self, _env_unset: None,
    ) -> None:
        """If multiple stages in the list need OpenRouter, the error
        must list every one of them (sorted) — otherwise the operator
        fixes one stage, hits the next, fixes that, and so on."""
        with pytest.raises(AxBenchBaseError) as exc_info:
            validate_env_for_stages(
                [Stage.JUDGE, Stage.SYNTH_POSITIVE_MODE],
            )
        msg = str(exc_info.value)
        assert "'1b'" in msg
        assert "'5'" in msg

    def test_error_message_includes_export_hint(
        self, _env_unset: None,
    ) -> None:
        """The error message tells the operator how to fix it."""
        with pytest.raises(AxBenchBaseError) as exc_info:
            validate_env_for_stages([Stage.JUDGE])
        assert "export" in str(exc_info.value)

    def test_error_message_triggering_stages_are_sorted(
        self, _env_unset: None,
    ) -> None:
        """Stage ids appear in sorted order so the error is
        deterministic across runs / input list orders."""
        with pytest.raises(AxBenchBaseError) as exc_info:
            validate_env_for_stages(
                [Stage.JUDGE, Stage.SYNTH_POSITIVE_MODE, Stage.SYNTH_NEGATIVE_MODE],
            )
        msg = str(exc_info.value)
        idx_1b = msg.find("'1b'")
        idx_1c = msg.find("'1c'")
        idx_5 = msg.find("'5'")
        assert idx_1b >= 0
        assert idx_1c >= 0
        assert idx_5 >= 0
        assert idx_1b < idx_1c < idx_5

    def test_error_message_deduplicates_triggering_stages(
        self, _env_unset: None,
    ) -> None:
        """Passing the same stage twice does not duplicate it in the
        error output."""
        with pytest.raises(AxBenchBaseError) as exc_info:
            validate_env_for_stages(
                [Stage.JUDGE, Stage.JUDGE, Stage.JUDGE],
            )
        msg = str(exc_info.value)
        assert msg.count("'5'") == 1


# ---------------------------------------------------------------------------
# Edge cases — cfg interactions
# ---------------------------------------------------------------------------


class TestCfgEdgeCases:
    def test_empty_string_env_var_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Item 16: an empty-string ``OPENROUTER_API_KEY`` is treated
        as unset. The validator reads the env var via
        ``os.environ.get(KEY)`` which returns ``""`` (falsy) for an
        empty export, so the ``if os.environ.get(...)`` check
        correctly raises -- operators do not accidentally pass
        validation with a blank key.
        """
        monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "")
        with pytest.raises(
            AxBenchBaseError, match=OPENROUTER_API_KEY_ENV,
        ):
            validate_env_for_stages([Stage.JUDGE])

    def test_whitespace_only_env_var_still_treated_as_set(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pinning a related edge case: a whitespace-only value is
        truthy to ``os.environ.get`` so validation passes. This is
        intentional -- we do not try to sanitize the key, we only
        check for presence. If the key is wrong the real API call
        will fail.
        """
        monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "  ")
        # No raise — whitespace is technically "set".
        validate_env_for_stages([Stage.JUDGE])

    def test_cfg_with_qc_enabled_stage_0c_and_env_unset_raises(
        self, _env_unset: None,
    ) -> None:
        """Explicit positive test of the QC-on Stage 0c default."""
        cfg = _make_cfg(enable_concept_examples_qc=True)
        with pytest.raises(AxBenchBaseError):
            validate_env_for_stages([Stage.PREPARE_NEGATIVE_PREFIXES], cfg=cfg)

    def test_cfg_with_qc_disabled_still_raises_for_other_stages(
        self, _env_unset: None, _np_env_set: None,
    ) -> None:
        """If the list contains Stage 0a + Stage 5, the helper
        raises ONLY for Stage 5 — the error must not include 0a."""
        cfg = _make_cfg(enable_concept_examples_qc=False)
        with pytest.raises(AxBenchBaseError) as exc_info:
            validate_env_for_stages(
                [Stage.PREPARE_CONCEPTS, Stage.JUDGE], cfg=cfg,
            )
        msg = str(exc_info.value)
        assert "'5'" in msg
        assert "'0a'" not in msg

    def test_stage_all_pseudo_not_checked_directly(
        self, _env_unset: None,
    ) -> None:
        """:attr:`Stage.ALL` is a pseudo-stage expanded by
        ``_resolve_stage_list``; passing it directly (without
        expansion) is a no-op because ``ALL`` is not a key in
        ``_STAGES_REQUIRING_OPENROUTER``. This is documented behavior —
        callers are expected to expand ``ALL`` before calling the
        validator."""
        # No raise.
        validate_env_for_stages([Stage.ALL])

    def test_cfg_none_for_prepare_negative_prefixes_is_defensive_default(
        self, _env_unset: None,
    ) -> None:
        """When ``cfg`` is :data:`None` and Stage 0c is requested, the
        helper assumes QC is on (the default) — the safer option."""
        with pytest.raises(AxBenchBaseError):
            validate_env_for_stages([Stage.PREPARE_NEGATIVE_PREFIXES], cfg=None)

    def test_cfg_none_for_data_stages_still_noop(
        self, _env_unset: None,
    ) -> None:
        """``cfg=None`` doesn't change the no-op behavior for pure data
        stages."""
        validate_env_for_stages(
            [Stage.AGGREGATE, Stage.GENERATE], cfg=None,
        )


# ---------------------------------------------------------------------------
# Parametrized sweep — every functional stage, raises/no-raises verified
# ---------------------------------------------------------------------------


#: Stages that are checked by the env validator (expect raise when env
#: is unset). Stage.PREPARE_CONCEPTS requires Neuronpedia; Stage
#: PREPARE_NEGATIVE_PREFIXES defensively requires examples-QC env when
#: ``cfg=None``.
_ENV_REQUIRED_STAGES: list[Stage] = [
    Stage.PREPARE_CONCEPTS,
    Stage.PREPARE_NEGATIVE_PREFIXES,
    Stage.SYNTH_POSITIVE_MODE,
    Stage.SYNTH_NEGATIVE_MODE,
    Stage.JUDGE,
]

#: Stages that the env validator ignores (expect no-op regardless of env).
_ENV_FREE_STAGES: list[Stage] = [
    Stage.PREPARE_SEED_POOLS,
    Stage.CONCEPT_FEATURE_MAPPING,
    Stage.TRAIN_DIFFMEAN,
    Stage.TRAIN_LINEAR_PROBE,
    Stage.TRAIN_PROMPTING,
    Stage.TRAIN_LORA,
    Stage.ENUMERATE_STAGE4_CONTEXTS,
    Stage.GENERATE,
    Stage.AGGREGATE,
]


class TestParametrizedSweep:
    """Enumerate every functional stage and pin env-validation behavior.

    Adding a new stage to :class:`Stage` without updating
    ``_STAGES_REQUIRING_OPENROUTER`` is a silent pitfall — this sweep
    makes that pitfall into an explicit test failure.
    """

    @pytest.mark.parametrize("stage", _ENV_REQUIRED_STAGES)
    def test_required_stage_raises_without_env(
        self, stage: Stage, _env_unset: None,
    ) -> None:
        with pytest.raises(AxBenchBaseError):
            validate_env_for_stages([stage])

    @pytest.mark.parametrize("stage", _ENV_REQUIRED_STAGES)
    def test_required_stage_passes_with_env(
        self, stage: Stage, _env_set: None,
    ) -> None:
        validate_env_for_stages([stage])

    @pytest.mark.parametrize("stage", _ENV_FREE_STAGES)
    def test_free_stage_is_noop_without_env(
        self, stage: Stage, _env_unset: None,
    ) -> None:
        validate_env_for_stages([stage])

    @pytest.mark.parametrize("stage", _ENV_FREE_STAGES)
    def test_free_stage_is_noop_with_env(
        self, stage: Stage, _env_set: None,
    ) -> None:
        validate_env_for_stages([stage])

    def test_required_and_free_partition_covers_every_functional_stage(
        self,
    ) -> None:
        """Regression: the two parametrized lists together must cover
        every member of :class:`Stage` except :attr:`Stage.ALL`, so
        future additions to :class:`Stage` force a test update."""
        functional = {s for s in Stage if s is not Stage.ALL}
        covered = set(_ENV_REQUIRED_STAGES) | set(_ENV_FREE_STAGES)
        assert covered == functional


# ---------------------------------------------------------------------------
# Regression tests on the exported symbol set
# ---------------------------------------------------------------------------


class TestExportedSymbolSet:
    def test_stages_requiring_openrouter_is_frozenset(self) -> None:
        """Immutability guard — the set is intended to be shared by
        reference across callers."""
        assert isinstance(_STAGES_REQUIRING_OPENROUTER, frozenset)

    def test_stages_requiring_openrouter_excludes_stage_0a_and_0c(self) -> None:
        """Stage 0a never uses OpenRouter; Stage 0c is handled
        conditionally via cfg because examples QC can be disabled."""
        assert Stage.PREPARE_CONCEPTS not in _STAGES_REQUIRING_OPENROUTER
        assert Stage.PREPARE_NEGATIVE_PREFIXES not in _STAGES_REQUIRING_OPENROUTER

    def test_stages_requiring_openrouter_contains_known_stages(self) -> None:
        """Pin the current membership so a silent removal shows up
        here immediately."""
        assert Stage.SYNTH_POSITIVE_MODE in _STAGES_REQUIRING_OPENROUTER
        assert Stage.SYNTH_NEGATIVE_MODE in _STAGES_REQUIRING_OPENROUTER
        assert Stage.JUDGE in _STAGES_REQUIRING_OPENROUTER

    def test_stages_requiring_openrouter_excludes_data_stages(self) -> None:
        """Stages 0b, 4, 6 must never require env."""
        assert Stage.PREPARE_SEED_POOLS not in _STAGES_REQUIRING_OPENROUTER
        assert Stage.GENERATE not in _STAGES_REQUIRING_OPENROUTER
        assert Stage.AGGREGATE not in _STAGES_REQUIRING_OPENROUTER


# ---------------------------------------------------------------------------
# Neuronpedia env validation (Stage 0a and lazy Stage 0c examples QC)
#
# Stage 0a needs Neuronpedia regardless of QC, so the validator must
# raise on a missing NEURONPEDIA_API_KEY even when QC is off (in which
# case the OpenRouter check would pass).
# ---------------------------------------------------------------------------


@pytest.fixture
def _np_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NEURONPEDIA_API_KEY_ENV, raising=False)


@pytest.fixture
def _np_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NEURONPEDIA_API_KEY_ENV, "np-test-value")


class TestNeuronpediaEnvValidation:
    def test_prepare_concepts_without_np_key_qc_off_raises(
        self, _env_set: None, _np_env_unset: None,
    ) -> None:
        """Stage 0a needs Neuronpedia even when QC is off."""
        cfg = _make_cfg(enable_concept_examples_qc=False)
        with pytest.raises(AxBenchBaseError, match=NEURONPEDIA_API_KEY_ENV):
            validate_env_for_stages([Stage.PREPARE_CONCEPTS], cfg=cfg)

    def test_prepare_concepts_without_np_key_qc_on_raises(
        self, _env_set: None, _np_env_unset: None,
    ) -> None:
        """Stage 0a + QC on still raises on missing Neuronpedia key."""
        cfg = _make_cfg(enable_concept_examples_qc=True)
        with pytest.raises(AxBenchBaseError, match=NEURONPEDIA_API_KEY_ENV):
            validate_env_for_stages([Stage.PREPARE_CONCEPTS], cfg=cfg)

    def test_prepare_concepts_with_both_keys_set_noop(
        self, _env_set: None, _np_env_set: None,
    ) -> None:
        cfg = _make_cfg(enable_concept_examples_qc=True)
        validate_env_for_stages([Stage.PREPARE_CONCEPTS], cfg=cfg)

    def test_prepare_negative_prefixes_without_np_key_qc_on_raises(
        self, _env_set: None, _np_env_unset: None,
    ) -> None:
        """Stage 0c examples QC fetches Neuronpedia activations lazily."""
        cfg = _make_cfg(enable_concept_examples_qc=True)
        with pytest.raises(AxBenchBaseError, match=NEURONPEDIA_API_KEY_ENV):
            validate_env_for_stages([Stage.PREPARE_NEGATIVE_PREFIXES], cfg=cfg)

    def test_prepare_negative_prefixes_qc_off_does_not_need_np_key(
        self, _env_set: None, _np_env_unset: None,
    ) -> None:
        cfg = _make_cfg(
            enable_concept_examples_qc=False,
            verification_api_model=None,
        )
        validate_env_for_stages([Stage.PREPARE_NEGATIVE_PREFIXES], cfg=cfg)

    def test_prepare_negative_prefixes_default_api_mode_requires_np_key(
        self, _env_set: None, _np_env_unset: None,
    ) -> None:
        cfg = _make_cfg(
            enable_concept_examples_qc=False,
        )
        assert cfg.verification_api_model == "inclusionai/ling-2.6-flash"
        with pytest.raises(AxBenchBaseError, match=NEURONPEDIA_API_KEY_ENV):
            validate_env_for_stages(
                [Stage.PREPARE_NEGATIVE_PREFIXES],
                cfg=cfg,
            )

    def test_other_stages_dont_require_np_key(
        self, _env_set: None, _np_env_unset: None,
    ) -> None:
        """Stages unrelated to Neuronpedia examples/descriptions ignore the key."""
        validate_env_for_stages([Stage.JUDGE])
        validate_env_for_stages([Stage.SYNTH_POSITIVE_MODE])
        validate_env_for_stages([Stage.AGGREGATE])

    def test_error_message_includes_export_hint(
        self, _env_set: None, _np_env_unset: None,
    ) -> None:
        with pytest.raises(AxBenchBaseError) as exc_info:
            validate_env_for_stages([Stage.PREPARE_CONCEPTS])
        assert "export" in str(exc_info.value)
        assert NEURONPEDIA_API_KEY_ENV in str(exc_info.value)

    def test_empty_string_np_key_treated_as_unset(
        self, _env_set: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(NEURONPEDIA_API_KEY_ENV, "")
        with pytest.raises(AxBenchBaseError, match=NEURONPEDIA_API_KEY_ENV):
            validate_env_for_stages([Stage.PREPARE_CONCEPTS])

    def test_both_envs_missing_error_lists_both(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If both keys are missing, the error message must mention both
        so the operator fixes them together rather than chasing one at
        a time."""
        monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
        monkeypatch.delenv(NEURONPEDIA_API_KEY_ENV, raising=False)
        cfg = _make_cfg(enable_concept_examples_qc=True)
        with pytest.raises(AxBenchBaseError) as exc_info:
            validate_env_for_stages([Stage.PREPARE_NEGATIVE_PREFIXES], cfg=cfg)
        msg = str(exc_info.value)
        assert OPENROUTER_API_KEY_ENV in msg
        assert NEURONPEDIA_API_KEY_ENV in msg
