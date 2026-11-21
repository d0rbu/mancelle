"""Pure prompt, replay-seed, and runtime contracts for SelfConditioning."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
from threading import enumerate as active_threads
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch as t

from nanochat.common import COMPUTE_DTYPE
from nanochat.tokenizer import SPECIAL_TOKENS
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondScoreInterpolation,
    SelfCondScoreMasking,
)
from steering.conditional.sampling import (
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
)
from steering.eval.axbench_adv import self_conditioning as self_conditioning_module
from steering.eval.axbench_adv.counter import CounterDirection
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import OptimizationExample, RawEvaluation
from steering.eval.axbench_adv.prompt_surgery import QWEN_CONTROL_TOKENS
from steering.eval.axbench_adv.self_conditioning import (
    SelfConditioningMethodRuntime,
    render_self_conditioning_prompt,
)
from steering.eval.axbench_upstream.data import ConceptTrainingTexts
from steering.eval.axbench_upstream.methods import (
    SELF_CONDITIONING_FACTORS,
    PreparedPromptContext,
)
from steering.eval.axbench_upstream.run_methods import (
    derive_generation_seed,
    derive_selection_seed,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt


class _Tokenizer:
    specials = {
        "<user_start>": 2,
        "<user_end>": 3,
        "<assistant_start>": 4,
        "<assistant_end>": 5,
    }

    def __init__(self, mutation: str | None = None) -> None:
        self.mutation = mutation
        self.conversations: list[object] = []

    def get_bos_token_id(self) -> int:
        return 1

    def encode_special(self, token: str) -> int:
        return self.specials[token]

    def encode(self, text: str) -> list[int]:
        return [byte + 10 for byte in text.encode("utf-8")]

    def decode(self, token_ids: list[int]) -> str:
        decoded = bytes(token_id - 10 for token_id in token_ids).decode("utf-8")
        return decoded + ("changed" if self.mutation == "decode" else "")

    def render_for_completion(self, conversation: object) -> list[int]:
        self.conversations.append(conversation)
        copied = deepcopy(conversation)
        assert isinstance(copied, dict)
        messages = copied.get("messages")
        assert isinstance(messages, list) and messages and isinstance(messages[0], dict)
        prompt = messages[0].get("content")
        assert isinstance(prompt, str)
        result = [1, 2, *self.encode(prompt), 3, 4]
        if self.mutation == "wrong_bos":
            result[0] = 9
        elif self.mutation == "missing":
            result.pop(-2)
        elif self.mutation == "duplicate":
            result.insert(-1, 3)
        elif self.mutation == "reordered":
            result[0], result[1] = result[1], result[0]
        elif self.mutation == "truncate":
            result = result[:5]
        return result


def test_raw_prompt_control_and_suffix_semantics_are_exact() -> None:
    tokenizer = _Tokenizer()
    original = "prompt with trailing whitespace \n"
    control, rendered_control = render_self_conditioning_prompt(
        tokenizer, original, " \n\t ", max_prompt_tokens=100
    )
    attacked, rendered_attacked = render_self_conditioning_prompt(
        tokenizer,
        original,
        "  suffix  with  internal\nspace  ",
        max_prompt_tokens=100,
    )
    unicode_attacked, _ = render_self_conditioning_prompt(
        tokenizer,
        "e\u0301 原文",
        "café \u0301",
        max_prompt_tokens=100,
    )

    assert control == original
    assert attacked == original + "\n\nsuffix  with  internal\nspace"
    assert unicode_attacked == "e\u0301 原文\n\ncafé \u0301"
    assert rendered_control.token_ids == [1, 2, *tokenizer.encode(original), 3, 4]
    assert rendered_attacked.token_ids == [1, 2, *tokenizer.encode(attacked), 3, 4]
    assert tokenizer.conversations[0] is not tokenizer.conversations[1]
    assert original == "prompt with trailing whitespace \n"


@pytest.mark.parametrize("value", ["bad\x00prompt", "\ud800", "\udfff"])
def test_raw_prompt_rejects_nul_and_surrogates(value: str) -> None:
    with pytest.raises(AxBenchAdvError, match="NUL|UTF-8"):
        render_self_conditioning_prompt(_Tokenizer(), value, "", max_prompt_tokens=20)


@pytest.mark.parametrize("value", ["bad\x00suffix", "\ud800", "\udfff"])
def test_suffix_rejects_nul_and_surrogates(value: str) -> None:
    with pytest.raises(AxBenchAdvError, match="NUL|UTF-8"):
        render_self_conditioning_prompt(_Tokenizer(), "prompt", value, max_prompt_tokens=30)


@pytest.mark.parametrize("token", QWEN_CONTROL_TOKENS)
def test_suffix_rejects_every_qwen_control_token(token: str) -> None:
    with pytest.raises(AxBenchAdvError, match="Qwen control"):
        render_self_conditioning_prompt(_Tokenizer(), "prompt", token, max_prompt_tokens=100)


@pytest.mark.parametrize(
    "token",
    [*SPECIAL_TOKENS, *(f"<|{token[1:-1]}|>" for token in SPECIAL_TOKENS)],
)
def test_suffix_rejects_current_and_legacy_nanochat_special_tokens(token: str) -> None:
    with pytest.raises(AxBenchAdvError, match="nanochat special"):
        render_self_conditioning_prompt(_Tokenizer(), "prompt", token, max_prompt_tokens=100)


def test_original_prompt_may_contain_special_token_looking_text() -> None:
    original = "Explain literal <assistant_start> and <|user_end|>."
    attacked, _ = render_self_conditioning_prompt(_Tokenizer(), original, "", max_prompt_tokens=100)
    assert attacked == original


@pytest.mark.parametrize(
    "mutation",
    ["wrong_bos", "missing", "duplicate", "reordered", "truncate"],
)
def test_render_structural_drift_and_silent_truncation_fail(mutation: str) -> None:
    with pytest.raises(AxBenchAdvError, match="structural sequence"):
        render_self_conditioning_prompt(
            _Tokenizer(mutation), "long enough prompt", "suffix", max_prompt_tokens=100
        )


def test_render_decode_mismatch_fails() -> None:
    with pytest.raises(AxBenchAdvError, match="decode exactly"):
        render_self_conditioning_prompt(
            _Tokenizer("decode"), "prompt", "suffix", max_prompt_tokens=100
        )


def test_decoder_prompt_budget_accepts_exact_limit_and_rejects_one_less() -> None:
    tokenizer = _Tokenizer()
    expected_length = len(tokenizer.encode("abc")) + 4
    render_self_conditioning_prompt(tokenizer, "abc", "", max_prompt_tokens=expected_length)
    with pytest.raises(AxBenchAdvError, match="decoder prompt budget"):
        render_self_conditioning_prompt(tokenizer, "abc", "", max_prompt_tokens=expected_length - 1)
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(AxBenchAdvError, match="max_prompt_tokens"):
            render_self_conditioning_prompt(
                tokenizer,
                "abc",
                "",
                max_prompt_tokens=invalid,  # ty: ignore[invalid-argument-type]
            )


@pytest.mark.parametrize("mutation", ["content-type", "content-element", "structure-type"])
def test_render_rejects_invalid_tokenizer_outputs(mutation: str) -> None:
    class InvalidTokenizer(_Tokenizer):
        def encode(self, text: str) -> Any:
            if mutation == "content-type":
                return (1, 2)
            if mutation == "content-element":
                return [True]
            return super().encode(text)

        def get_bos_token_id(self) -> Any:
            return "bos" if mutation == "structure-type" else super().get_bos_token_id()

    with pytest.raises(AxBenchAdvError, match="tokenizer returned invalid"):
        render_self_conditioning_prompt(InvalidTokenizer(), "prompt", "", max_prompt_tokens=100)


def test_prompt_rng_requires_shared_cpu_sampling_bundle() -> None:
    rng = t.Generator(device="cpu").manual_seed(1)
    selection = SamplingBundle(SamplingSelection(), t.zeros(2), rng)
    negative = NegativeBundle(NegativeSamplingSelection(), 2, rng)
    setup = SimpleNamespace(
        prompt_pipeline=SimpleNamespace(
            selection_config=selection,
            negative_bundle=negative,
        )
    )
    assert (
        self_conditioning_module._prompt_rng(
            setup  # ty: ignore[invalid-argument-type]
        )
        is rng
    )

    setup.prompt_pipeline.selection_config = object()
    with pytest.raises(AxBenchAdvError, match="SamplingBundle"):
        self_conditioning_module._prompt_rng(setup)  # ty: ignore[invalid-argument-type]
    setup.prompt_pipeline.selection_config = selection
    setup.prompt_pipeline.negative_bundle = None
    with pytest.raises(AxBenchAdvError, match="negative sampling"):
        self_conditioning_module._prompt_rng(setup)  # ty: ignore[invalid-argument-type]
    setup.prompt_pipeline.negative_bundle = NegativeBundle(
        NegativeSamplingSelection(), 2, t.Generator(device="cpu")
    )
    with pytest.raises(AxBenchAdvError, match="share one generator"):
        self_conditioning_module._prompt_rng(setup)  # ty: ignore[invalid-argument-type]

    fake_rng = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    fake_selection = object.__new__(SamplingBundle)
    fake_selection.rng = fake_rng  # ty: ignore[invalid-assignment]
    setup.prompt_pipeline.selection_config = fake_selection
    setup.prompt_pipeline.negative_bundle = SimpleNamespace(rng=fake_rng)
    with pytest.raises(AxBenchAdvError, match="must be on CPU"):
        self_conditioning_module._prompt_rng(setup)  # ty: ignore[invalid-argument-type]


def test_source_prompt_and_prepared_context_validators_fail_closed() -> None:
    with pytest.raises(AxBenchAdvError, match="missing required"):
        self_conditioning_module._source_prompts(
            object(),  # ty: ignore[invalid-argument-type]
            max_concepts=2,
        )
    frame = _runtime_frame()
    with pytest.raises(AxBenchAdvError, match="canonical key product"):
        self_conditioning_module._source_prompts(frame.iloc[:-1], max_concepts=2)
    varying_prompt = frame.copy()
    varying_prompt.loc[0, "original_prompt"] = "different"
    with pytest.raises(AxBenchAdvError, match="original_prompt varies"):
        self_conditioning_module._source_prompts(varying_prompt, max_concepts=2)
    varying_concept = frame.copy()
    varying_concept.loc[0, "input_concept"] = "different"
    with pytest.raises(AxBenchAdvError, match="concept text varies across source factors"):
        self_conditioning_module._source_prompts(varying_concept, max_concepts=2)
    across_inputs = frame.copy()
    across_inputs.loc[across_inputs["input_id"].eq(1), "input_concept"] = "different"
    with pytest.raises(AxBenchAdvError, match="concept text varies across inputs"):
        self_conditioning_module._source_prompts(across_inputs, max_concepts=2)

    setup = SimpleNamespace(n_sae_features=10, feature_vocab=None)
    empty = PreparedPromptContext((1,), 2, ())
    with pytest.raises(AxBenchAdvError, match="no real feature segments"):
        self_conditioning_module._validate_prepared(
            setup,  # ty: ignore[invalid-argument-type]
            empty,
            label="empty",
        )
    mismatched = PreparedPromptContext((1,), 2, (ContextFeatureSegment((1,), (0, 1), 0, 1),))
    with pytest.raises(AxBenchAdvError, match="mismatched feature arrays"):
        self_conditioning_module._validate_prepared(
            setup,  # ty: ignore[invalid-argument-type]
            mismatched,
            label="bad",
        )
    duplicate = PreparedPromptContext((1,), 2, (ContextFeatureSegment((1, 1), (0, 1), 0, 1),))
    with pytest.raises(AxBenchAdvError, match="duplicate feature ids"):
        self_conditioning_module._validate_prepared(
            setup,  # ty: ignore[invalid-argument-type]
            duplicate,
            label="bad",
        )
    valid = PreparedPromptContext((1,), 2, (ContextFeatureSegment((1, 2), (0, 1), 0, 1),))
    self_conditioning_module._validate_prepared(
        setup,  # ty: ignore[invalid-argument-type]
        valid,
        label="valid",
    )


@pytest.mark.parametrize(
    "meta",
    [
        {},
        {"model_config": None},
        {"model_config": {"sequence_len": True}},
        {"model_config": {"sequence_len": 0}},
    ],
)
def test_metadata_sequence_length_is_strict(meta: dict[str, Any]) -> None:
    with pytest.raises(AxBenchAdvError, match="sequence_len"):
        self_conditioning_module._metadata_sequence_len(
            SimpleNamespace(meta=meta)  # ty: ignore[invalid-argument-type]
        )


def test_historical_seed_streams_have_exact_golden_vectors() -> None:
    assert derive_selection_seed(42, 0, 0) == 448_567_669
    assert derive_selection_seed(42, 3, 7) == 1_184_360_016
    assert derive_generation_seed(42, "SelfConditioning", 0, 0, 6.0) == 1_685_524_595
    assert derive_generation_seed(42, "SelfConditioning", 3, 7, 6.0) == 180_211_800
    assert derive_generation_seed(42, "SelfConditioning", 3, 7, -0.0) == 49_288_835
    assert derive_generation_seed(42, "SelfConditioning", 3, 7, 0.0) == 974_542_769


def test_seed_domains_reproduce_and_change_only_declared_dimensions() -> None:
    selection = derive_selection_seed(42, 3, 7)
    token = derive_generation_seed(42, "SelfConditioning", 3, 7, 6.0)
    assert derive_selection_seed(42, 3, 7) == selection
    assert derive_generation_seed(42, "SelfConditioning", 3, 7, 6.0) == token
    assert derive_selection_seed(43, 3, 7) != selection
    assert derive_generation_seed(43, "SelfConditioning", 3, 7, 6.0) != token
    assert derive_selection_seed(42, 4, 7) != selection
    assert derive_selection_seed(42, 3, 8) != selection
    assert derive_generation_seed(42, "SelfConditioning", 3, 7, 5.0) != token
    assert derive_selection_seed(42, 3, 7) == selection


@pytest.mark.parametrize("value", [True, -1, 1 << 63, "1", 1.5, None])
def test_seed_integer_domains_reject_invalid_values(value: Any) -> None:
    for function, args in (
        (derive_selection_seed, (value, 0, 0)),
        (derive_selection_seed, (0, value, 0)),
        (derive_selection_seed, (0, 0, value)),
        (derive_generation_seed, (value, "SelfConditioning", 0, 0, 1.0)),
        (derive_generation_seed, (0, "SelfConditioning", value, 0, 1.0)),
        (derive_generation_seed, (0, "SelfConditioning", 0, value, 1.0)),
    ):
        with pytest.raises(ValueError, match="integer"):
            dynamic_function: Any = function
            dynamic_function(*args)


@pytest.mark.parametrize("factor", [True, -1.0, float("nan"), float("inf"), "1", None])
def test_generation_seed_rejects_invalid_factors(factor: Any) -> None:
    with pytest.raises(ValueError, match="factor"):
        derive_generation_seed(42, "SelfConditioning", 0, 0, factor)


def test_generation_seed_rejects_method_extension_surface() -> None:
    with pytest.raises(ValueError, match="SelfConditioning"):
        derive_generation_seed(42, "SelfConditioning/suffix", 0, 0, 1.0)


def _runtime_frame(*, duplicate_prompts: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "concept_id": concept_id,
                "input_id": input_id,
                "factor": factor,
                "input_concept": f"concept-{concept_id}",
                "original_prompt": (
                    "shared prompt" if duplicate_prompts else f"raw prompt {concept_id}/{input_id}"
                ),
            }
            for concept_id in range(2)
            for input_id in range(10)
            for factor in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
        ]
    )


def _training() -> dict[int, ConceptTrainingTexts]:
    return {
        concept_id: ConceptTrainingTexts(
            concept_id,
            f"concept-{concept_id}",
            "genre",
            (f"positive-{concept_id}",),
            (f"negative-{concept_id}",),
        )
        for concept_id in range(2)
    }


class _Judge:
    def __init__(self, payload: dict[str, Any], events: list[tuple[Any, ...]]) -> None:
        self.payload = payload
        self.events = events
        self.closed = False

    def evaluate(self, *, generation: str, input_id: int) -> RawEvaluation:
        self.events.append(("judge", generation, input_id))
        return RawEvaluation(generation, 1.0, 1.0, "{}", 0)

    def close(self) -> None:
        self.events.append(("judge-close", self.payload["concept_id"]))
        self.closed = True

    def rebind(self, payload: dict[str, Any]) -> None:
        self.events.append(("judge-rebind", self.payload["method"], payload["method"]))
        self.payload = payload

    def abort(self) -> None:
        self.events.append(("judge-abort", self.payload["concept_id"]))
        self.closed = True


def _runtime_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    duplicate_prompts: bool = False,
    target_failure: bool = False,
    event_log: list[tuple[Any, ...]] | None = None,
    target_sequence_len: int = 1024,
    runtime_updates: dict[str, Any] | None = None,
    judge_factory_failure: bool = False,
    judge_factory_failure_at: int | None = None,
    judge_close_failure: bool = False,
    judge_changes_generation: bool = False,
    infeasible_concept: int | None = None,
    fitted_concepts: list[tuple[int, ...]] | None = None,
) -> tuple[SelfConditioningMethodRuntime, list[tuple[Any, ...]], t.Generator]:
    events: list[tuple[Any, ...]] = [] if event_log is None else event_log
    rng = t.Generator(device="cpu").manual_seed(1234)
    setup = SimpleNamespace(
        meta={"model_config": {"sequence_len": 1024}},
        output_negative_selection=NegativeSamplingSelection(),
    )
    target = SimpleNamespace(dec_config=SimpleNamespace(sequence_len=target_sequence_len))

    monkeypatch.setattr(self_conditioning_module, "_prompt_rng", lambda _setup: rng)
    monkeypatch.setattr(self_conditioning_module, "get_tokenizer", lambda: _Tokenizer())
    monkeypatch.setattr(
        self_conditioning_module,
        "load_extraction_stack",
        lambda passed, device: (
            events.append(("load-extraction", passed is setup, device)) or object(),
            object(),
        ),
    )

    def prepare(
        passed: object,
        _model: object,
        _sae: object,
        _tokenizer: object,
        *,
        token_ids: list[int],
        assistant_end_id: int,
    ) -> PreparedPromptContext:
        draw = int(t.randint(0, 1_000_000, (), generator=rng).item())
        events.append(("prepare", passed is setup, tuple(token_ids), draw))
        return PreparedPromptContext(tuple(token_ids), assistant_end_id, ())

    monkeypatch.setattr(self_conditioning_module, "prepare_prompt_context", prepare)
    monkeypatch.setattr(
        self_conditioning_module, "_validate_prepared", lambda *_args, **_kwargs: None
    )
    def fit(
        passed: object, _model: object, _sae: object, training: dict[int, Any], **_kwargs: Any
    ) -> dict[int, t.Tensor]:
        events.append(("fit", passed is setup))
        if fitted_concepts is not None:
            fitted_concepts.append(tuple(training))
        return {concept_id: t.tensor([float(concept_id)]) for concept_id in training}

    monkeypatch.setattr(self_conditioning_module, "fit_concept_dense_diffmeans", fit)

    def assert_feasible(
        passed: object,
        scores: t.Tensor,
        *,
        concept_id: int,
    ) -> None:
        events.append(
            (
                "preflight",
                passed is setup,
                scores.clone(),
                concept_id,
            )
        )
        if concept_id == infeasible_concept:
            raise ValueError(f"concept {concept_id} is infeasible")

    monkeypatch.setattr(
        self_conditioning_module,
        "assert_masked_diffmean_feasible",
        assert_feasible,
    )

    masked_policy_factors: dict[int, float] = {}

    def build_masked_policy(
        passed: object,
        scores: t.Tensor,
        factor: float,
    ) -> SelfCondScoreMasking:
        events.append(("build-masked-policy", passed is setup, factor))
        policy = SelfCondScoreMasking(
            target_scores=scores,
            negative_selection=setup.output_negative_selection,
            positive_pool_limit=PositiveInt.parse(round(factor * 10) + 1),
            negative_pool_limit=PositiveInt.parse(round(factor * 10) + 2),
        )
        masked_policy_factors[id(policy)] = factor
        return policy

    monkeypatch.setattr(
        self_conditioning_module,
        "build_self_conditioning_masked_policy",
        build_masked_policy,
    )

    def load_target(passed: object, device: str) -> tuple[object, object]:
        events.append(("load-target", passed is setup, device))
        if target_failure:
            raise RuntimeError("target load failed")
        return target, object()

    monkeypatch.setattr(self_conditioning_module, "load_target_model", load_target)
    monkeypatch.setattr(
        self_conditioning_module,
        "assert_prepared_prompt_matches",
        lambda _tokenizer, raw, prepared: events.append(("parity", raw, prepared.token_ids)),
    )

    def generate(
        passed: object,
        loaded_target: object,
        _tokenizer: object,
        prepared: PreparedPromptContext,
        policy: SelfCondScoreInterpolation | SelfCondScoreMasking,
        *,
        selection_seed: int,
        seed: int,
        max_new_tokens: int,
    ) -> str:
        if isinstance(policy, SelfCondScoreInterpolation):
            method = "SelfConditioning"
            factor = min(
                SELF_CONDITIONING_FACTORS,
                key=lambda candidate: abs(candidate / 6.0 - float(policy.weight)),
            )
        else:
            assert isinstance(policy, SelfCondScoreMasking)
            method = "SelfConditioningMasked"
            factor = masked_policy_factors[id(policy)]
        generation = f"generated:{prepared.token_ids[-5:]}:{factor}:{selection_seed}:{seed}"
        events.append(
            (
                "generate",
                passed is setup,
                loaded_target is target,
                prepared,
                factor,
                selection_seed,
                seed,
                max_new_tokens,
                generation,
                method,
                policy,
            )
        )
        return generation

    monkeypatch.setattr(self_conditioning_module, "generate_self_conditioning", generate)

    def judge_factory(*, init_payload: dict[str, Any], **_kwargs: Any) -> _Judge:
        events.append(("judge-open", init_payload, _kwargs))
        if judge_factory_failure or (
            judge_factory_failure_at is not None
            and [event[0] for event in events].count("judge-open") == judge_factory_failure_at
        ):
            raise RuntimeError("judge open failed")
        judge = _Judge(init_payload, events)
        if judge_close_failure:
            judge.close = (  # ty: ignore[invalid-assignment]
                lambda: (_ for _ in ()).throw(RuntimeError("judge close failed"))
            )
        if judge_changes_generation:
            judge.evaluate = (  # ty: ignore[invalid-assignment]
                lambda *, generation, input_id: RawEvaluation(
                    generation + " changed", 1.0, 1.0, "{}", 0
                )
            )
        return judge

    runtime_args: dict[str, Any] = {
        "setup": setup,
        "training": _training(),
        "main_frame": _runtime_frame(duplicate_prompts=duplicate_prompts),
        "max_concepts": 2,
        "max_new_tokens": 128,
        "master_seed": 42,
        "device": "cpu",
        "extraction_batch_size": 8,
        "judge_factory": judge_factory,
        "judge_cache": tmp_path / "judge.sqlite",
        "judge_model": "fake/judge",
        "api_base_url": "http://localhost:8000/v1/",
        "cache_only": True,
        "timeout_seconds": 1.0,
        "stderr_dir": tmp_path / "logs",
    }
    runtime_args.update(runtime_updates or {})
    runtime = SelfConditioningMethodRuntime(
        **runtime_args,
    )
    return runtime, events, rng


def _examples(concept_id: int, *, duplicate_prompts: bool = False) -> list[OptimizationExample]:
    return [
        OptimizationExample(
            input_id,
            "shared prompt" if duplicate_prompts else f"raw prompt {concept_id}/{input_id}",
        )
        for input_id in range(10)
    ]


def test_selected_fit_keeps_full_source_validation_and_original_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fitted: list[tuple[int, ...]] = []
    runtime, _events, _rng = _runtime_harness(
        monkeypatch,
        tmp_path,
        runtime_updates={"fit_concept_ids": (1,)},
        fitted_concepts=fitted,
    )
    assert fitted == [(1,)]
    assert set(runtime._training) == {0, 1}
    assert set(runtime._dense_diffmeans) == {1}
    with pytest.raises(AxBenchAdvError, match="no fitted DiffMean"):
        runtime.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        )
    runtime.close()


@pytest.mark.parametrize("invalid", [(), (2,), (True,), (1, 1), (1.0,)])
def test_selected_fit_rejects_invalid_source_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, invalid: tuple[object, ...]
) -> None:
    with pytest.raises(AxBenchAdvError, match="fit_concept_ids"):
        _runtime_harness(monkeypatch, tmp_path, runtime_updates={"fit_concept_ids": invalid})


def test_native_extraction_routes_to_weight_free_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = SimpleNamespace(
        meta={"model_config": {"sequence_len": 1024}},
        output_negative_selection=NegativeSamplingSelection(),
        sae_model="gemma-3-4b-pt",
        sae_layer=22,
        resolved_sae=SimpleNamespace(hook_name="blocks.22.hook_resid_post"),
    )
    seen: list[dict[str, object]] = []

    def load(**kwargs: object) -> tuple[object, object]:
        seen.append(kwargs)
        return object(), object()

    monkeypatch.setattr(
        self_conditioning_module, "load_weight_free_gemma3_extraction_stack", load
    )
    runtime, events, _rng = _runtime_harness(
        monkeypatch,
        tmp_path,
        runtime_updates={
            "setup": setup,
            "device": "cuda:0",
            "weight_free_extraction": True,
            "native_extraction": True,
        },
    )
    assert seen == [{
        "sae_model": "gemma-3-4b-pt",
        "resolved_sae": setup.resolved_sae,
        "device": "cuda:0",
        "native": True,
        "sae_layer": 22,
    }]
    assert sum(event[0] == "prepare" for event in events) == 20
    runtime.close()

    for options in (
        {"native_extraction": True},
        {"native_extraction": True, "weight_free_extraction": True},
        {"native_extraction": 1},
    ):
        with pytest.raises(AxBenchAdvError, match="native_extraction"):
            _runtime_harness(monkeypatch, tmp_path, runtime_updates=options)


def test_project_pair_concurrency_is_only_sent_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, _events, _rng = _runtime_harness(monkeypatch, tmp_path)
    payload = runtime._judge_payload(
        method="SelfConditioning", concept_id=0, concept="concept-0", factor=6.0
    )
    assert "pair_concurrency" not in payload
    runtime.close()

    runtime, _events, _rng = _runtime_harness(
        monkeypatch, tmp_path, runtime_updates={"judge_pair_concurrency": 2}
    )
    payload = runtime._judge_payload(
        method="SelfConditioning", concept_id=0, concept="concept-0", factor=6.0
    )
    assert payload["pair_concurrency"] == 2
    runtime.close()

    with pytest.raises(AxBenchAdvError, match="judge_pair_concurrency"):
        _runtime_harness(monkeypatch, tmp_path, runtime_updates={"judge_pair_concurrency": True})


@pytest.mark.parametrize("inflight,workers", [(2, 1), (3, 2)])
def test_judge_overlap_advances_generation_with_bounded_backpressure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inflight: int, workers: int
) -> None:
    runtime, events, _rng = _runtime_harness(
        monkeypatch,
        tmp_path,
        runtime_updates={"judge_inflight": inflight, "judge_workers": workers},
    )
    judge_started = Event()
    second_generated = Event()
    release_judge = Event()
    original_generate = self_conditioning_module.generate_self_conditioning
    original_judge = _Judge.evaluate
    generations = 0

    def generate(*args: Any, **kwargs: Any) -> str:
        nonlocal generations
        result = original_generate(*args, **kwargs)
        generations += 1
        if generations == inflight:
            second_generated.set()
        return result

    def judge(self: _Judge, *, generation: str, input_id: int) -> RawEvaluation:
        if input_id == 0:
            judge_started.set()
            assert release_judge.wait(5), "slow judge was never released"
        return original_judge(self, generation=generation, input_id=input_id)

    monkeypatch.setattr(self_conditioning_module, "generate_self_conditioning", generate)
    monkeypatch.setattr(_Judge, "evaluate", judge)
    with (
        runtime,
        runtime.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ) as cell,
    ):
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(cell.evaluate, "", list(range(inflight + 1)))
            try:
                assert judge_started.wait(5)
                assert second_generated.wait(5)
                assert generations == inflight
                assert not result.done()
            finally:
                release_judge.set()
            assert len(result.result(timeout=5)) == inflight + 1
    assert generations == inflight + 1
    assert sorted(event[2] for event in events if event[0] == "judge") == list(range(inflight + 1))


@pytest.mark.parametrize("workers,inflight", [(1, 2), (2, 2), (2, 3)])
@pytest.mark.parametrize("method", ["SelfConditioning", "SelfConditioningMasked", "NLSteering"])
def test_judge_overlap_preserves_counter_results_and_seed_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str, workers: int, inflight: int
) -> None:
    runtime, _events, _rng = _runtime_harness(
        monkeypatch,
        tmp_path,
        runtime_updates={"judge_inflight": inflight, "judge_workers": workers},
    )
    runtime._nl_target_scores = {0: t.ones(8)}
    target_model = SimpleNamespace(
        dec_config=SimpleNamespace(n_layer=3, n_embd=2),
        training=False,
        _post_cross_attention_eval_callback=None,
        get_device=lambda: t.device("cpu"),
    )
    runtime._target_model = target_model
    with (
        runtime,
        runtime.bind_cell(
            method=method,
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ) as cell,
    ):
        runtime._judge_inflight = 1
        serial_control = cell.evaluate("", list(range(10)))
        runtime._judge_inflight = inflight
        assert cell.evaluate("", list(range(10))) == serial_control
        for direction in (t.tensor([1.0, -2.0]), t.tensor([-3.0, 4.0])):
            counter = CounterDirection(direction, {"layer": 1})
            runtime._judge_inflight = 1
            serial = cell.evaluate_counter(counter, 0.5, list(range(10)))
            runtime._judge_inflight = inflight
            assert cell.evaluate_counter(counter, 0.5, list(range(10))) == serial
            assert target_model._post_cross_attention_eval_callback is None


@pytest.mark.parametrize("workers,inflight", [(1, 2), (2, 2), (2, 3)])
@pytest.mark.parametrize("failure", ["worker_death", "interrupt"])
def test_judge_overlap_failure_drains_and_cleans_counter_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str, workers: int, inflight: int
) -> None:
    runtime, events, _rng = _runtime_harness(
        monkeypatch,
        tmp_path,
        runtime_updates={"judge_inflight": inflight, "judge_workers": workers},
    )
    target_model = SimpleNamespace(
        dec_config=SimpleNamespace(n_layer=3, n_embd=2),
        training=False,
        _post_cross_attention_eval_callback=None,
        get_device=lambda: t.device("cpu"),
    )
    runtime._target_model = target_model
    if failure == "worker_death":

        def dead_judge(self: _Judge, *, generation: str, input_id: int) -> RawEvaluation:
            del self, generation, input_id
            raise AxBenchAdvError("worker protocol failure")

        monkeypatch.setattr(_Judge, "evaluate", dead_judge)
        expected = AxBenchAdvError
    else:
        original_generate = self_conditioning_module.generate_self_conditioning
        calls = 0

        def interrupted_generate(*args: Any, **kwargs: Any) -> str:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return original_generate(*args, **kwargs)

        monkeypatch.setattr(
            self_conditioning_module, "generate_self_conditioning", interrupted_generate
        )
        expected = KeyboardInterrupt
    with pytest.raises(expected):
        with (
            runtime,
            runtime.bind_cell(
                method="SelfConditioning",
                concept_id=0,
                concept="concept-0",
                factor=6.0,
                examples=_examples(0),
            ) as cell,
        ):
            cell.evaluate_counter(
                CounterDirection(t.tensor([1.0, 0.0]), {"layer": 1}), 0.5, [0, 1, 2]
            )
    assert target_model._post_cross_attention_eval_callback is None
    assert [event[0] for event in events].count("judge-close") == workers
    assert not any(thread.name.startswith("axbench-adv-judge") for thread in active_threads())


def test_two_judge_workers_run_concurrently_and_return_in_input_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, events, _rng = _runtime_harness(
        monkeypatch, tmp_path, runtime_updates={"judge_inflight": 2, "judge_workers": 2}
    )
    started = [Event(), Event()]
    release = Event()
    original_judge = _Judge.evaluate

    def judge(self: _Judge, *, generation: str, input_id: int) -> RawEvaluation:
        if input_id < 2:
            started[input_id].set()
            assert release.wait(5), "both judge workers were not released"
        return original_judge(self, generation=generation, input_id=input_id)

    monkeypatch.setattr(_Judge, "evaluate", judge)
    with (
        runtime,
        runtime.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ) as cell,
    ):
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(cell.evaluate, "", [0, 1, 2])
            try:
                assert all(event.wait(5) for event in started)
            finally:
                release.set()
            parallel = result.result(timeout=5)
        runtime._judge_inflight = 1
        assert parallel == cell.evaluate("", [0, 1, 2])
    assert sorted(event[2] for event in events if event[0] == "judge") == [0, 0, 1, 1, 2, 2]
    assert [event[0] for event in events].count("judge-close") == 2


def test_second_judge_worker_startup_failure_closes_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, events, _rng = _runtime_harness(
        monkeypatch, tmp_path, runtime_updates={"judge_inflight": 2, "judge_workers": 2}
    )
    original_factory = runtime._judge_factory
    calls = 0

    def fail_second(**kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second judge startup failed")
        return original_factory(**kwargs)

    runtime._judge_factory = fail_second
    with runtime, pytest.raises(RuntimeError, match="second judge startup failed"):
        with runtime.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ):
            pass
    assert [event[0] for event in events].count("judge-close") == 1
    assert runtime._active_cell is None


def test_judge_pool_rebinds_between_methods_and_concepts_then_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, events, _rng = _runtime_harness(
        monkeypatch,
        tmp_path,
        runtime_updates={"judge_inflight": 3, "judge_workers": 2, "reuse_judge_workers": True},
    )
    with runtime:
        for method, concept_id in (
            ("SelfConditioning", 0),
            ("SelfConditioningMasked", 0),
            ("SelfConditioning", 1),
        ):
            with runtime.bind_cell(
                method=method,
                concept_id=concept_id,
                concept=f"concept-{concept_id}",
                factor=6.0,
                examples=_examples(concept_id),
            ) as cell:
                assert len(cell.evaluate("", [0, 1])) == 2
            assert not any(event[0] == "judge-close" for event in events)
    assert [event[0] for event in events].count("judge-open") == 2
    assert [event[0] for event in events].count("judge-rebind") == 4
    assert [event[0] for event in events].count("judge-close") == 2
    assert runtime._judge_pool == ()


def test_judge_pool_prewarms_before_first_cell_and_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, events, _rng = _runtime_harness(
        monkeypatch,
        tmp_path,
        runtime_updates={
            "judge_inflight": 3,
            "judge_workers": 2,
            "reuse_judge_workers": True,
            "prewarm_judge_cell": ("SelfConditioning", 0, "concept-0", 6.0),
        },
    )
    assert len(runtime._judge_pool) == 2
    assert [event[0] for event in events].count("judge-open") == 2
    with runtime:
        with runtime.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ) as cell:
            assert len(cell.evaluate("", [0])) == 1
    assert [event[0] for event in events].count("judge-rebind") == 2
    assert [event[0] for event in events].count("judge-close") == 2


def test_judge_pool_prewarm_aborts_on_runtime_init_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[tuple[Any, ...]] = []
    with pytest.raises(RuntimeError, match="target load failed"):
        _runtime_harness(
            monkeypatch,
            tmp_path,
            target_failure=True,
            event_log=events,
            runtime_updates={
                "judge_inflight": 3,
                "judge_workers": 2,
                "reuse_judge_workers": True,
                "prewarm_judge_cell": ("SelfConditioning", 0, "concept-0", 6.0),
            },
        )
    assert [event[0] for event in events].count("judge-open") == 2
    assert [event[0] for event in events].count("judge-abort") == 2


def test_judge_pool_prewarm_aborts_first_worker_if_second_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[tuple[Any, ...]] = []
    with pytest.raises(RuntimeError, match="judge open failed"):
        _runtime_harness(
            monkeypatch,
            tmp_path,
            event_log=events,
            judge_factory_failure_at=2,
            runtime_updates={
                "judge_inflight": 3,
                "judge_workers": 2,
                "reuse_judge_workers": True,
                "prewarm_judge_cell": ("SelfConditioning", 0, "concept-0", 6.0),
            },
        )
    assert [event[0] for event in events].count("judge-open") == 2
    assert [event[0] for event in events].count("judge-abort") == 1


def test_judge_pool_aborts_all_workers_on_rebind_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, events, _rng = _runtime_harness(
        monkeypatch,
        tmp_path,
        runtime_updates={"judge_inflight": 3, "judge_workers": 2, "reuse_judge_workers": True},
    )
    with runtime:
        with runtime.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ):
            pass
        runtime._judge_pool[1].rebind = (  # ty: ignore[invalid-assignment]
            lambda _payload: (_ for _ in ()).throw(RuntimeError("rebind failed"))
        )
        with pytest.raises(RuntimeError, match="rebind failed"):
            with runtime.bind_cell(
                method="SelfConditioningMasked",
                concept_id=0,
                concept="concept-0",
                factor=6.0,
                examples=_examples(0),
            ):
                pass
        assert runtime._judge_pool == ()
    assert [event[0] for event in events].count("judge-abort") == 2


def test_judge_pool_aborts_all_workers_on_cell_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, events, _rng = _runtime_harness(
        monkeypatch,
        tmp_path,
        runtime_updates={"judge_inflight": 3, "judge_workers": 2, "reuse_judge_workers": True},
    )
    with runtime, pytest.raises(RuntimeError, match="cell failed"):
        with runtime.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ):
            raise RuntimeError("cell failed")
    assert runtime._judge_pool == ()
    assert runtime._active_cell is None
    assert [event[0] for event in events].count("judge-abort") == 2


def test_judge_pool_requires_strict_bool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(AxBenchAdvError, match="reuse_judge_workers"):
        _runtime_harness(monkeypatch, tmp_path, runtime_updates={"reuse_judge_workers": 1})


@pytest.mark.parametrize("workers,inflight", [(1, 2), (2, 2), (2, 3)])
def test_judge_overlap_keeps_duplicate_cache_accounting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workers: int, inflight: int
) -> None:
    runtime, _events, _rng = _runtime_harness(
        monkeypatch,
        tmp_path,
        runtime_updates={"judge_inflight": inflight, "judge_workers": workers},
    )
    cache: set[str] = set()
    judge_inputs: list[int] = []

    def cached_judge(self: _Judge, *, generation: str, input_id: int) -> RawEvaluation:
        del self
        judge_inputs.append(input_id)
        paid = 0 if generation in cache else 2
        cache.add(generation)
        return RawEvaluation(generation, 1.0, 1.5, "{}", paid)

    monkeypatch.setattr(_Judge, "evaluate", cached_judge)
    monkeypatch.setattr(
        self_conditioning_module,
        "generate_self_conditioning",
        lambda *_args, **_kwargs: "duplicate generation",
    )
    with (
        runtime,
        runtime.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ) as cell,
    ):
        runtime._judge_inflight = 1
        serial = cell.evaluate("", [0, 1, 2])
        cache.clear()
        runtime._judge_inflight = inflight
        overlapped = cell.evaluate("", [0, 1, 2])
    assert overlapped == serial
    assert [row.paid_judge_cache_misses for row in overlapped] == [2, 0, 0]
    assert judge_inputs == [0, 1, 2, 0, 1, 2]


def test_runtime_load_order_replay_cache_seeds_and_judge_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, events, _rng = _runtime_harness(monkeypatch, tmp_path)
    event_names = [event[0] for event in events]  # type: ignore[index]
    assert event_names.count("load-extraction") == 1
    assert event_names.count("prepare") == 20
    assert event_names.count("fit") == 1
    assert event_names.count("preflight") == 2
    assert event_names.count("load-target") == 1
    assert event_names.index("load-extraction") < event_names.index("prepare")
    assert max(i for i, name in enumerate(event_names) if name == "prepare") < event_names.index(
        "fit"
    )
    assert event_names.index("fit") < event_names.index("preflight")
    assert max(i for i, name in enumerate(event_names) if name == "preflight") < event_names.index(
        "load-target"
    )
    preflights = [event for event in events if event[0] == "preflight"]
    assert [event[3] for event in preflights] == [0, 1]
    assert event_names.count("parity") == 20

    with runtime:
        with runtime.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ) as cell:
            before = sum(event[0] == "prepare" for event in events)  # type: ignore[index]
            control = cell.evaluate("", [2, 1])
            assert [raw.generation for raw in control]
            assert sum(event[0] == "prepare" for event in events) == before  # type: ignore[index]
            first = cell.evaluate("  suffix  ", [2, 1])
            after_cold = sum(event[0] == "prepare" for event in events)  # type: ignore[index]
            second = cell.evaluate("suffix", [2, 1])
            assert sum(event[0] == "prepare" for event in events) == after_cold  # type: ignore[index]
            assert first == second
            generate_events = [event for event in events if event[0] == "generate"]
            assert len(generate_events) == 6
            for event, input_id in zip(generate_events[-2:], (2, 1), strict=True):
                assert event[5] == derive_selection_seed(42, 0, input_id)
                assert event[6] == derive_generation_seed(42, "SelfConditioning", 0, input_id, 6.0)
                assert event[7] == 128
            judge_events = [event for event in events if event[0] == "judge"]
            assert [event[2] for event in judge_events] == [2, 1, 2, 1, 2, 1]
            assert [event[1] for event in judge_events] == [event[8] for event in generate_events]
        assert not cell._cache  # ty: ignore[unresolved-attribute]
    assert runtime._closed  # type: ignore[attr-defined]
    payload = next(event[1] for event in events if event[0] == "judge-open")
    assert payload == {
        "api_base_url": "http://localhost:8000/v1",
        "cache_only": True,
        "concept": "concept-0",
        "concept_id": 0,
        "factor": 6.0,
        "judge_cache": str((tmp_path / "judge.sqlite").resolve()),
        "judge_model": "fake/judge",
        "method": "SelfConditioning",
        "runtime_kind": "judge",
    }


def test_duplicate_original_tokens_prepare_once_and_share_saved_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, events, _rng = _runtime_harness(monkeypatch, tmp_path, duplicate_prompts=True)
    assert sum(event[0] == "prepare" for event in events) == 1  # type: ignore[index]
    states = [context.rng_state for context in runtime._originals.values()]  # type: ignore[attr-defined]
    assert all(t.equal(states[0], state) for state in states[1:])
    runtime.close()


def test_root_method_dispatch_rank_policies_seeds_identity_and_cell_local_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, events, _rng = _runtime_harness(monkeypatch, tmp_path)
    selection_seeds: list[int] = []
    token_seeds: list[int] = []
    with runtime:
        for method, factor in (
            ("SelfConditioning", 6.0),
            ("SelfConditioningMasked", 0.0),
            ("SelfConditioningMasked", 6.0),
        ):
            with runtime.bind_cell(
                method=method,
                concept_id=0,
                concept="concept-0",
                factor=factor,
                examples=_examples(0),
            ) as cell:
                cell.evaluate("same suffix", [0])
                event = next(item for item in reversed(events) if item[0] == "generate")
                selection_seeds.append(event[5])
                token_seeds.append(event[6])
                assert event[9] == method
                policy = event[10]
                if method == "SelfConditioning":
                    assert isinstance(policy, SelfCondScoreInterpolation)
                    assert float(policy.weight) == 1.0
                else:
                    assert isinstance(policy, SelfCondScoreMasking)
                    assert int(policy.positive_pool_limit) == round(factor * 10) + 1
                    assert int(policy.negative_pool_limit) == round(factor * 10) + 2

    assert selection_seeds == [derive_selection_seed(42, 0, 0)] * 3
    assert len(set(token_seeds)) == 3
    suffix_prepares = [
        event
        for event in events
        if event[0] == "prepare" and len(event[2]) > len(_Tokenizer().encode("raw prompt 0/0")) + 4
    ]
    assert len(suffix_prepares) == 3
    judge_opens = [event for event in events if event[0] == "judge-open"]
    assert [event[1]["method"] for event in judge_opens] == [
        "SelfConditioning",
        "SelfConditioningMasked",
        "SelfConditioningMasked",
    ]
    assert [event[2]["stderr_path"] for event in judge_opens] == [
        (tmp_path / "logs/SelfConditioning/c0.log").resolve(),
        (tmp_path / "logs/SelfConditioningMasked/c0.log").resolve(),
        (tmp_path / "logs/SelfConditioningMasked/c0.log").resolve(),
    ]


def test_failed_candidate_does_not_cache_or_perturb_following_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, events, rng = _runtime_harness(monkeypatch, tmp_path)
    real_prepare = self_conditioning_module.prepare_prompt_context

    def failing_prepare(*args: Any, token_ids: list[int], **kwargs: Any) -> PreparedPromptContext:
        needle = _Tokenizer().encode("fail")
        if any(token_ids[index : index + len(needle)] == needle for index in range(len(token_ids))):
            t.randint(0, 1_000_000, (), generator=rng)
            raise RuntimeError("candidate extraction failed")
        return real_prepare(*args, token_ids=token_ids, **kwargs)

    monkeypatch.setattr(self_conditioning_module, "prepare_prompt_context", failing_prepare)
    with runtime:
        with runtime.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ) as cell:
            with pytest.raises(RuntimeError, match="candidate extraction failed"):
                cell.evaluate("fail", [0])
            assert (0, "fail") not in cell._cache  # ty: ignore[unresolved-attribute]
            successful = cell.evaluate("ok", [0])
            cell._cache.pop((0, "ok"))  # ty: ignore[unresolved-attribute]
            repeated = cell.evaluate("ok", [0])
            assert successful == repeated
            draws = [event[3] for event in events if event[0] == "prepare"]
            assert draws[-1] == draws[-2]


def test_runtime_initialization_failure_cleans_loaded_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[Any, ...]] = []
    with pytest.raises(RuntimeError, match="target load failed"):
        _runtime_harness(
            monkeypatch,
            tmp_path,
            target_failure=True,
            event_log=events,
        )
    assert [event[0] for event in events].count("load-extraction") == 1  # type: ignore[index]
    assert [event[0] for event in events].count("load-target") == 1  # type: ignore[index]


def test_infeasible_masked_pool_prevents_target_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="concept 1 is infeasible"):
        _runtime_harness(
            monkeypatch,
            tmp_path,
            event_log=events,
            infeasible_concept=1,
        )
    assert [event[0] for event in events].count("preflight") == 2  # type: ignore[index]
    assert not any(event[0] == "load-target" for event in events)  # type: ignore[index]


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"max_concepts": True}, "max_concepts"),
        ({"training": {0: _training()[0]}}, "training concepts"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"master_seed": True}, "master_seed"),
        ({"master_seed": -1}, "master_seed"),
        ({"judge_inflight": True}, "judge_inflight"),
        ({"judge_inflight": 4}, "judge_inflight"),
        ({"judge_inflight": 3}, "judge_inflight"),
        ({"judge_inflight": 1, "judge_workers": 2}, "judge_workers"),
        ({"extraction_batch_size": True}, "extraction_batch_size"),
        ({"extraction_batch_size": 0}, "extraction_batch_size"),
        ({"max_new_tokens": 1024}, "exhausts decoder context"),
        (
            {
                "training": {
                    0: ConceptTrainingTexts(0, "wrong", "genre", ("p",), ("n",)),
                    1: _training()[1],
                }
            },
            "training concept differs",
        ),
    ],
)
def test_runtime_rejects_invalid_initialization_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    updates: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(AxBenchAdvError, match=match):
        _runtime_harness(monkeypatch, tmp_path, runtime_updates=updates)


def test_runtime_rejects_loaded_decoder_sequence_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(AxBenchAdvError, match="decoder sequence length"):
        _runtime_harness(monkeypatch, tmp_path, target_sequence_len=2048)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"method": "DiffMean"}, "unsupported root method"),
        ({"concept_id": True}, "concept_id"),
        ({"concept": "wrong"}, "concept text"),
        ({"factor": True}, "factor must be numeric"),
        ({"factor": 7.0}, "declared grid"),
        ({"factor": float("nan")}, "declared grid"),
        ({"examples": [object()] * 10}, "OptimizationExample"),
        ({"examples": _examples(0)[:-1]}, "cover input ids"),
        (
            {
                "examples": [
                    OptimizationExample(
                        input_id, "wrong" if input_id == 0 else f"raw prompt 0/{input_id}"
                    )
                    for input_id in range(10)
                ]
            },
            "example text",
        ),
    ],
)
def test_bind_cell_rejects_invalid_identity_and_examples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    updates: dict[str, Any],
    match: str,
) -> None:
    runtime, _events, _rng = _runtime_harness(monkeypatch, tmp_path)
    arguments: dict[str, Any] = {
        "method": "SelfConditioning",
        "concept_id": 0,
        "concept": "concept-0",
        "factor": 6.0,
        "examples": _examples(0),
    }
    arguments.update(updates)
    with pytest.raises(AxBenchAdvError, match=match):
        runtime.bind_cell(**arguments)
    if updates.get("method") == "DiffMean":
        with pytest.raises(AxBenchAdvError, match="unsupported root method"):
            self_conditioning_module._BoundSelfConditioningCell(
                runtime,
                **arguments,
            )
        with pytest.raises(AxBenchAdvError, match="unsupported root method"):
            runtime._judge_payload(  # type: ignore[attr-defined]
                method="DiffMean",
                concept_id=0,
                concept="concept-0",
                factor=6.0,
            )
    runtime.close()


def test_cell_active_request_and_judge_identity_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, _events, _rng = _runtime_harness(monkeypatch, tmp_path)
    cell = runtime.bind_cell(
        method="SelfConditioning",
        concept_id=0,
        concept="concept-0",
        factor=6.0,
        examples=_examples(0),
    )
    with pytest.raises(AxBenchAdvError, match="not active"):
        cell.evaluate("", [0])  # ty: ignore[unresolved-attribute]
    with runtime, cell:
        second = runtime.bind_cell(
            method="SelfConditioning",
            concept_id=1,
            concept="concept-1",
            factor=6.0,
            examples=_examples(1),
        )
        with pytest.raises(AxBenchAdvError, match="only one"):
            second.__enter__()
        with pytest.raises(AxBenchAdvError, match="integers in 0..9"):
            cell.evaluate("", [10])  # ty: ignore[unresolved-attribute]
    cell.close()  # ty: ignore[unresolved-attribute]
    with pytest.raises(AxBenchAdvError, match="closed"):
        runtime.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        )
    with pytest.raises(AxBenchAdvError, match="closed"):
        runtime.__enter__()

    changed, _events, _rng = _runtime_harness(
        monkeypatch, tmp_path / "changed", judge_changes_generation=True
    )
    with changed:
        with changed.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ) as changed_cell:
            with pytest.raises(AxBenchAdvError, match="changed the generated text"):
                changed_cell.evaluate("", [0])


def test_judge_open_and_close_failures_clear_cache_and_preserve_primary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    open_failure, _events, _rng = _runtime_harness(
        monkeypatch, tmp_path / "open", judge_factory_failure=True
    )
    cell = open_failure.bind_cell(
        method="SelfConditioning",
        concept_id=0,
        concept="concept-0",
        factor=6.0,
        examples=_examples(0),
    )
    with pytest.raises(RuntimeError, match="judge open failed"):
        cell.__enter__()
    assert not cell._cache  # ty: ignore[unresolved-attribute]
    cell.close()  # ty: ignore[unresolved-attribute]
    open_failure.close()

    close_failure, _events, _rng = _runtime_harness(
        monkeypatch, tmp_path / "close", judge_close_failure=True
    )
    cell = close_failure.bind_cell(
        method="SelfConditioning",
        concept_id=0,
        concept="concept-0",
        factor=6.0,
        examples=_examples(0),
    )
    cell.__enter__()
    with pytest.raises(RuntimeError, match="judge close failed"):
        cell.close()  # ty: ignore[unresolved-attribute]
    cell.close()  # ty: ignore[unresolved-attribute]
    close_failure.close()

    annotated, _events, _rng = _runtime_harness(
        monkeypatch, tmp_path / "annotated", judge_close_failure=True
    )
    with pytest.raises(RuntimeError, match="primary") as exc_info:
        with annotated:
            with annotated.bind_cell(
                method="SelfConditioning",
                concept_id=0,
                concept="concept-0",
                factor=6.0,
                examples=_examples(0),
            ):
                raise RuntimeError("primary")
    assert "cell cleanup failed" in "\n".join(exc_info.value.__notes__)

    unhandled, _events, _rng = _runtime_harness(
        monkeypatch, tmp_path / "unhandled", judge_close_failure=True
    )
    with pytest.raises(RuntimeError, match="judge close failed"):
        with unhandled.bind_cell(
            method="SelfConditioning",
            concept_id=0,
            concept="concept-0",
            factor=6.0,
            examples=_examples(0),
        ):
            pass
    unhandled.close()


def _bare_runtime(device: str = "cpu") -> SelfConditioningMethodRuntime:
    runtime = object.__new__(SelfConditioningMethodRuntime)
    runtime._active_cell = None  # type: ignore[attr-defined]
    runtime._judge_pool = ()
    runtime._target_model = object()  # type: ignore[attr-defined]
    runtime._target_tokenizer = object()  # type: ignore[attr-defined]
    runtime._extraction_model = object()  # type: ignore[attr-defined]
    runtime._sae = object()  # type: ignore[attr-defined]
    runtime._nl_target_scores = {}
    runtime._dense_diffmeans = {0: t.tensor([1.0])}  # type: ignore[attr-defined]
    runtime._originals = {"key": object()}  # ty: ignore[invalid-assignment]
    runtime._device = device  # type: ignore[attr-defined]
    runtime._closed = False  # type: ignore[attr-defined]
    return runtime


def test_runtime_cleanup_success_cuda_and_active_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runtime = _bare_runtime("cuda:0")
    runtime._active_cell = SimpleNamespace(close=lambda: events.append("cell-close"))  # ty: ignore[invalid-assignment]
    monkeypatch.setattr(self_conditioning_module.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(self_conditioning_module.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        self_conditioning_module.t.cuda, "empty_cache", lambda: events.append("cuda")
    )
    runtime.close()
    runtime.close()
    assert events == ["cell-close", "gc", "cuda"]
    assert runtime._active_cell is None  # type: ignore[attr-defined]
    assert runtime._target_model is None  # type: ignore[attr-defined]
    assert not runtime._dense_diffmeans  # type: ignore[attr-defined]
    assert not runtime._originals  # type: ignore[attr-defined]


def test_runtime_cleanup_failures_preserve_primary_and_raise_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cell() -> None:
        raise RuntimeError("cell cleanup")

    def fail_gc() -> None:
        raise RuntimeError("gc cleanup")

    runtime = _bare_runtime()
    runtime._active_cell = SimpleNamespace(close=fail_cell)  # ty: ignore[invalid-assignment]
    monkeypatch.setattr(self_conditioning_module.gc, "collect", fail_gc)
    primary = ValueError("primary")
    runtime._cleanup(primary=primary)  # type: ignore[attr-defined]
    assert len(primary.__notes__) == 2

    runtime = _bare_runtime()
    runtime._active_cell = SimpleNamespace(close=fail_cell)  # ty: ignore[invalid-assignment]
    with pytest.raises(RuntimeError, match="cell cleanup") as exc_info:
        runtime.close()
    assert "additional SelfConditioning cleanup failure" in "\n".join(exc_info.value.__notes__)


def test_runtime_exit_preserves_primary_when_cleanup_fails() -> None:
    runtime = _bare_runtime()
    runtime.close = lambda: (_ for _ in ()).throw(RuntimeError("cleanup"))  # ty: ignore[invalid-assignment]
    primary = ValueError("primary")
    runtime.__exit__(ValueError, primary, None)
    assert "runtime cleanup failed" in "\n".join(primary.__notes__)
    with pytest.raises(RuntimeError, match="cleanup"):
        runtime.__exit__(None, None, None)


@pytest.mark.parametrize("method", ["SelfConditioning", "SelfConditioningMasked", "NLSteering"])
def test_project_counter_scopes_post_cross_attention_callback_and_preserves_plain_evaluation(
    monkeypatch, tmp_path, method
):
    runtime, _events, _rng = _runtime_harness(monkeypatch, tmp_path)
    target_model = SimpleNamespace(
        dec_config=SimpleNamespace(n_layer=3, n_embd=2),
        training=False,
        _post_cross_attention_eval_callback=None,
        get_device=lambda: t.device("cpu"),
    )
    runtime._target_model = target_model
    runtime._nl_target_scores = {0: t.ones(8)}
    direction = t.tensor([2.0, -4.0])
    counter = CounterDirection(direction, {"layer": 1})
    observed = []

    def generate(*_args, **_kwargs):
        value = t.ones(1, 3, 2, dtype=COMPUTE_DTYPE)
        callback = target_model._post_cross_attention_eval_callback
        layer_values = [callback(index, value) if callback else value for index in range(3)]
        observed.append(layer_values[1])
        t.testing.assert_close(layer_values[0], value, rtol=0, atol=0)
        t.testing.assert_close(layer_values[2], value, rtol=0, atol=0)
        return "generation"

    monkeypatch.setattr(self_conditioning_module, "generate_self_conditioning", generate)
    with (
        runtime,
        runtime.bind_cell(
            method=method, concept_id=0, concept="concept-0", factor=6.0, examples=_examples(0)
        ) as cell,
    ):
        cell.evaluate("", [0])
        cell.evaluate_counter(counter, 0.5, [0])
        cell.evaluate("", [0])
    expected = observed[0] - 0.5 * direction.to(COMPUTE_DTYPE)
    t.testing.assert_close(observed[1], expected, rtol=0, atol=0)
    t.testing.assert_close(observed[2], observed[0], rtol=0, atol=0)
    assert target_model._post_cross_attention_eval_callback is None
