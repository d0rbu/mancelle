"""The checkpoint's RustBPE cadence must reach the generated refresh boundaries.

Nothing else spans this chain: ``load_method_setup`` resolves the cadence from
post-training metadata, ``prepare_prompt_context`` builds the turn prompt entry
with the live conditioned-SFT selector, ``build_self_conditioning_config`` turns
the cadence into ``SelfCondConfig.self_cond_segment_size``, and only then does
``generate_stream`` publish a new conditioning state every K assistant inputs.
The unit suites each pin one link with a hand-built K; this file pins that the
K the runtime uses is the checkpoint's, not ``model.dec_config.sequence_len``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.eval.axbench_upstream import methods
from steering.eval.axbench_upstream.encdec_baselines import generation as encdec_generation
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.target_conditioning import SingleEntryContextProvider
from steering.segmentation import SegmentMode
from tests.conftest import MockTokenizer
from tests.encdec_test_helpers import N_SAE_FEATURES, build_tiny_model
from tests.scripts.eval.test_axbench_project_methods import (
    _install_turn_setup_fixture,
    _turn_prompt_pipeline,
)

_CADENCE = 3


def _turn_target_model() -> Any:
    """A tiny encoder-decoder with the MAX_POOL head SelfConditioning needs."""

    return build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        feature_pred_config=FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
    )


def _loaded_turn_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Any:
    """Resolve a real ``MethodSetup`` from turn metadata carrying the cadence."""

    fixture = _install_turn_setup_fixture(
        monkeypatch,
        tmp_path,
        real_selection_context=True,
        prompt_pipeline=_turn_prompt_pipeline(seed=5),
    )
    meta = fixture.meta_path.read_text(encoding="utf-8").replace(
        '"turn_self_cond_segment_size_rustbpe": 37',
        f'"turn_self_cond_segment_size_rustbpe": {_CADENCE}',
    )
    fixture.meta_path.write_text(meta, encoding="utf-8")
    return methods.load_method_setup(fixture.meta_path, fixture.statistics_path)


def _tiny_bench_setup(loaded: Any) -> Any:
    """Keep the loaded turn provenance; swap in the tiny model's SAE geometry.

    ``load_method_setup`` pins the production Gemma-3-4B/65k feature contract,
    which no CPU test model can carry. Only the feature-space fields are
    replaced; ``segment_mode``, the RustBPE cadence and the live selection
    context all come from the real load.
    """

    return replace(
        loaded,
        n_sae_features=int(N_SAE_FEATURES),
        percentile_table=t.zeros((int(N_SAE_FEATURES), 99), dtype=t.float64),
        feature_contract=SimpleNamespace(feature_mask=None, consolidation=None),
        consolidation=None,
        prompt_consolidation=None,
    )


def _turn_prepared_prompt(
    monkeypatch: pytest.MonkeyPatch,
    setup: Any,
    tokenizer: MockTokenizer,
) -> Any:
    """Run the real turn selector and converter over one rendered prompt."""

    prompt_ids = tuple(
        tokenizer.render_for_completion(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": ""},
                ]
            }
        )
    )
    rendered_text, turn_starts = methods.rendered_text_and_turn_char_starts(
        prompt_ids,
        tokenizer,
    )
    bos_text = "<bos>"
    starts = [0, *(len(bos_text) + start for start in turn_starts[1:])]
    ends = [*starts[1:], len(bos_text) + len(rendered_text)]
    n_segments = len(starts)

    class _GemmaTokenizer:
        bos_token_id = 1

        def decode(self, ids: list[int]) -> str:
            assert ids == [1]
            return bos_text

    monkeypatch.setattr(methods, "truncate_to_training_window", lambda *_args: ([1], 3))
    monkeypatch.setattr(
        methods,
        "extract_document_features",
        lambda **_kwargs: (
            [t.tensor([index, index + 1], dtype=t.int64) for index in range(n_segments)],
            [t.tensor([8.0, 1.0], dtype=t.float64) for _ in range(n_segments)],
            list(range(n_segments)),
            starts,
            ends,
            3,
        ),
    )
    return methods.prepare_prompt_context(
        setup,
        SimpleNamespace(tokenizer=_GemmaTokenizer(), cfg=SimpleNamespace(n_ctx=8192)),
        object(),
        tokenizer,
        token_ids=prompt_ids,
        assistant_end_id=tokenizer.encode_special("<assistant_end>"),
    )


def _record_provider_steps(monkeypatch: pytest.MonkeyPatch, engine: Any) -> list[int]:
    """Record the number of buffered FP rows handed to each provider publish."""

    provider = engine._model._self_cond.provider
    real_step = provider.step
    widths: list[int] = []

    def recording_step(state: Any, completed_rows: t.Tensor) -> Any:
        widths.append(int(completed_rows.shape[1]))
        return real_step(state, completed_rows)

    monkeypatch.setattr(provider, "step", recording_step)
    return widths


def test_checkpoint_cadence_drives_the_generated_refresh_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """K comes from the checkpoint, and publishes land every K inputs."""

    # kills: using model.dec_config.sequence_len as the turn self-cond window
    # (the refresh would never fire inside the budget), and any change that
    # makes the runtime publish on a cadence other than the checkpoint's.
    tokenizer = MockTokenizer()
    model = _turn_target_model()
    model.eval()
    loaded = _loaded_turn_setup(monkeypatch, tmp_path)

    assert loaded.segment_mode is SegmentMode.turn_delimited_fixed_tokens
    assert loaded.turn_self_cond_segment_size_rustbpe == _CADENCE

    setup = _tiny_bench_setup(loaded)
    prepared = _turn_prepared_prompt(monkeypatch, setup, tokenizer)
    assert prepared.token_ids[-1] == tokenizer.encode_special("<assistant_start>")

    config = methods.build_self_conditioning_config(
        setup,
        model,
        None,
        selection_seed=17,
    )

    assert config.segment_mode is SegmentMode.turn_delimited_fixed_tokens
    assert int(config.self_cond_segment_size) == _CADENCE
    assert int(config.self_cond_segment_size) != int(model.dec_config.sequence_len)

    engine = build_engine_self_conditioned(
        model,
        tokenizer,
        t.device("cpu"),
        config,
        context_provider=SingleEntryContextProvider(
            prepared.token_ids,
            prepared.segments,
            segment_mode=prepared.segment_mode,
            token_assignments=prepared.token_assignments,
        ),
    )
    widths = _record_provider_steps(monkeypatch, engine)
    max_new_tokens = 3 * _CADENCE
    scheduled = iter(range(10, 10 + max_new_tokens))
    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        lambda *_args: t.tensor([[next(scheduled)]], dtype=t.long),
    )
    snapshots: list[tuple[int, int]] = []

    generated = []
    for token in engine.generate_stream(
        list(prepared.token_ids),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=None,
        seed=None,
        stop_token_ids=set(),
    ):
        generated.append(token)
        snapshot = engine._model.self_cond_runtime_snapshot()
        snapshots.append((len(generated), snapshot.tokens_in_seg))

    assert len(generated) == max_new_tokens
    # Every publish consumed exactly K feature-prediction rows -- one per
    # assistant input in the completed window.
    assert widths == [_CADENCE] * 3
    # A snapshot taken right after yielding token N reflects the marker plus
    # the N-1 generated inputs forwarded so far. The first snapshot is
    # therefore (1, 1): <assistant_start> alone, occupying input row 1.
    assert snapshots[0] == (1, 1)
    assert [tokens for _count, tokens in snapshots] == [
        (index + 1) % _CADENCE for index in range(max_new_tokens)
    ]
    boundary_positions = [count for count, tokens in snapshots if tokens == 0]
    assert boundary_positions == [_CADENCE, 2 * _CADENCE, 3 * _CADENCE]
    assert len(boundary_positions) >= 2


def test_encdec_baseline_scalar_turn_path_uses_the_same_checkpoint_cadence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The declared encdec-baseline scalar turn surface uses the same K."""

    # kills: an encdec-baseline turn cell that falls back to the fixed-mode
    # sequence_len window (a supported turn surface would then refresh on the
    # wrong geometry with no error).
    tokenizer = MockTokenizer()
    model = _turn_target_model()
    model.eval()
    loaded = _loaded_turn_setup(monkeypatch, tmp_path)
    setup = _tiny_bench_setup(loaded)
    prepared = _turn_prepared_prompt(monkeypatch, setup, tokenizer)
    observed: dict[str, Any] = {}
    # generate_encdec_baseline delegates to methods._generate_self_conditioning,
    # so the engine the encdec surface actually runs is built there.
    real_build_engine = methods.build_engine_self_conditioned

    def recording_build_engine(model_arg: Any, *args: Any, **kwargs: Any) -> Any:
        engine = real_build_engine(model_arg, *args, **kwargs)
        observed["config"] = args[2]
        observed["widths"] = _record_provider_steps(monkeypatch, engine)
        return engine

    monkeypatch.setattr(methods, "build_engine_self_conditioned", recording_build_engine)
    scheduled = iter(range(10, 10 + 3 * _CADENCE))
    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        lambda *_args: t.tensor([[next(scheduled)]], dtype=t.long),
    )

    text = encdec_generation.generate_encdec_baseline(
        setup,
        model,
        tokenizer,
        prepared,
        selection_seed=17,
        seed=3,
        max_new_tokens=2 * _CADENCE,
    )

    assert isinstance(text, str)
    assert observed["config"].segment_mode is SegmentMode.turn_delimited_fixed_tokens
    assert int(observed["config"].self_cond_segment_size) == _CADENCE
    assert observed["widths"] == [_CADENCE, _CADENCE]
