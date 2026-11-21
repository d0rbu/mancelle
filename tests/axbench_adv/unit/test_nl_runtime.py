"""Optional retrieval uses the established prompt, interpolation and seed path."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t

from steering.conditional.encoder_decoder.self_conditioned import SelfCondScoreInterpolation
from steering.conditional.sampling import NegativeSamplingSelection
from steering.eval.axbench_adv import self_conditioning as runtime_module
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_upstream.seeds import derive_generation_seed, derive_selection_seed
from steering.refined_types import UnitInterval
from tests.axbench_adv.unit.test_self_conditioning import _examples, _runtime_harness


def _setup() -> SimpleNamespace:
    return SimpleNamespace(
        n_sae_features=4, feature_vocab=None,
        meta={"model_config": {"sequence_len": 1024}},
        output_negative_selection=NegativeSamplingSelection(),
    )


@pytest.mark.parametrize("factor", [0.0, 0.2, 1.0, 3.0, 6.0])
def test_nl_cell_dispatches_exact_policy_and_replays_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factor: float,
) -> None:
    # The harness mocks generation: this verifies dispatch, policy and seed parity;
    # real-checkpoint CUDA generation equality remains a separate acceptance gate.
    scores = {0: t.tensor([-1.0, 0.0, 0.25, 1.0]), 1: t.zeros(4)}
    expected = scores[0].clone()
    runtime, events, _rng = _runtime_harness(
        monkeypatch, tmp_path, runtime_updates={"setup": _setup(), "nl_target_scores": scores},
    )
    scores[0].zero_()  # Caller mutation cannot change an active runtime's target.
    with runtime:
        with runtime.bind_cell(
            method="NLSteering", concept_id=0, concept="concept-0", factor=factor,
            examples=_examples(0),
        ) as cell:
            control = cell.evaluate("", [2])
            first = cell.evaluate("  avoid this topic  ", [2])
            before = sum(event[0] == "prepare" for event in events)
            replay = cell.evaluate("avoid this topic", [2])
            assert first == replay
            assert sum(event[0] == "prepare" for event in events) == before
            for event in [event for event in events if event[0] == "generate"]:
                policy = event[-1]
                assert isinstance(policy, SelfCondScoreInterpolation)
                assert t.equal(policy.target_scores, expected)
                assert float(policy.weight) == factor / 6.0
                assert event[5] == derive_selection_seed(42, 0, 2)
                assert event[6] == derive_generation_seed(42, "NLSteering", 0, 2, factor)
            prepared = runtime._originals[(0, 2)].prepared
            direct = runtime_module.generate_self_conditioning(
                runtime._setup, runtime._target_model, runtime._target_tokenizer, prepared,
                SelfCondScoreInterpolation(
                    target_scores=expected, negative_selection=runtime._setup.output_negative_selection,
                    weight=UnitInterval.parse(factor / 6.0),
                ),
                selection_seed=derive_selection_seed(42, 0, 2),
                seed=derive_generation_seed(42, "NLSteering", 0, 2, factor),
                max_new_tokens=128,
            )
            assert control[0].generation == direct
    payload = next(event[1] for event in events if event[0] == "judge-open")
    assert payload["method"] == "NLSteering"
    assert not runtime._nl_target_scores


@pytest.mark.parametrize("malformed", [
    {}, {0: t.zeros(4)}, {0: t.zeros(3), 1: t.zeros(3)},
    {0: t.full((4,), float("nan")), 1: t.zeros(4)},
    {0: t.full((4,), 1.01), 1: t.zeros(4)},
    {0: t.zeros(4, dtype=t.float64), 1: t.zeros(4)},
])
def test_invalid_nl_targets_fail_before_loading_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: dict[int, t.Tensor],
) -> None:
    events: list[tuple] = []
    with pytest.raises(AxBenchAdvError, match="NLSteering target tensors"):
        _runtime_harness(
            monkeypatch, tmp_path, event_log=events,
            runtime_updates={"setup": _setup(), "nl_target_scores": malformed},
        )
    assert not any(event[0].startswith("load-") for event in events)


def test_disabled_runtime_rejects_nl_and_keeps_original_seed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _events, _rng = _runtime_harness(monkeypatch, tmp_path)
    with runtime, pytest.raises(AxBenchAdvError, match="not enabled"):
        runtime.bind_cell(
            method="NLSteering", concept_id=0, concept="concept-0", factor=6.0,
            examples=_examples(0),
        )
    # Independent stable seed oracle applies to old and optional labels alike.
    for method in ("SelfConditioning", "SelfConditioningMasked", "NLSteering"):
        fields = ("42", method, "0", "2", (3.0).hex())
        expected = int.from_bytes(hashlib.sha256("\x1f".join(fields).encode()).digest()[:8], "big") % (2**31 - 1)
        assert derive_generation_seed(42, method, 0, 2, 3.0) == expected


@pytest.mark.parametrize("width", [2, 4])
def test_nl_runtime_uses_compacted_head_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int,
) -> None:
    setup = _setup()
    setup.feature_vocab = SimpleNamespace(size=2)
    scores = {0: t.zeros(width), 1: t.zeros(width)}
    events: list[tuple] = []
    if width == 4:
        with pytest.raises(AxBenchAdvError, match="target tensors"):
            _runtime_harness(monkeypatch, tmp_path, event_log=events, runtime_updates={
                "setup": setup, "nl_target_scores": scores,
            })
        assert not any(event[0].startswith("load-") for event in events)
    else:
        runtime, _events, _rng = _runtime_harness(monkeypatch, tmp_path, runtime_updates={
            "setup": setup, "nl_target_scores": scores,
        })
        with runtime:
            assert runtime._nl_target_scores[0].shape == (2,)
