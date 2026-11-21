from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st
from omegaconf import OmegaConf

import scripts.eval.dump_chat_responses as dump_module
from steering.eval.target_conditioning import TargetConditioningMode
from steering.refined_types import NonNegativeInt, PositiveInt

_PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
_NN_0 = NonNegativeInt.parse(0)
_NN_1 = NonNegativeInt.parse(1)
_NN_2 = NonNegativeInt.parse(2)
_NN_3 = NonNegativeInt.parse(3)
_NN_7 = NonNegativeInt.parse(7)
_POS_1 = PositiveInt.parse(1)
_POS_3 = PositiveInt.parse(3)

_INVALID_NON_NEGATIVE = st.one_of(
    st.integers(max_value=-1),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.none(),
)
_INVALID_POSITIVE = st.one_of(st.integers(max_value=0), _INVALID_NON_NEGATIVE)


def test_iter_local_indices_stripes_by_rank() -> None:
    assert list(
        dump_module.iter_local_indices(_NN_7, rank=_NN_0, world_size=_POS_3)
    ) == [0, 3, 6]
    assert list(
        dump_module.iter_local_indices(_NN_7, rank=_NN_1, world_size=_POS_3)
    ) == [1, 4]
    assert list(
        dump_module.iter_local_indices(_NN_7, rank=_NN_2, world_size=_POS_3)
    ) == [2, 5]


def test_iter_local_indices_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="0 <= rank < world_size"):
        dump_module.iter_local_indices(_NN_3, rank=_NN_3, world_size=_POS_3)


@_PROPERTY_SETTINGS
@given(
    n=st.integers(min_value=0, max_value=100),
    rank=st.integers(min_value=0, max_value=20),
    world_size=st.integers(min_value=1, max_value=20),
)
def test_iter_local_indices_matches_range_over_valid_scalar_domains(
    n: int,
    rank: int,
    world_size: int,
) -> None:
    refined_n = NonNegativeInt.parse(n)
    refined_rank = NonNegativeInt.parse(rank)
    refined_world_size = PositiveInt.parse(world_size)
    if rank >= world_size:
        with pytest.raises(ValueError, match="0 <= rank < world_size"):
            dump_module.iter_local_indices(
                refined_n,
                rank=refined_rank,
                world_size=refined_world_size,
            )
    else:
        assert list(
            dump_module.iter_local_indices(
                refined_n,
                rank=refined_rank,
                world_size=refined_world_size,
            )
        ) == list(range(rank, n, world_size))


@_PROPERTY_SETTINGS
@given(
    st.one_of(
        st.tuples(st.just("n"), _INVALID_NON_NEGATIVE),
        st.tuples(st.just("rank"), _INVALID_NON_NEGATIVE),
        st.tuples(st.just("world_size"), _INVALID_POSITIVE),
    )
)
def test_iter_local_indices_rejects_complete_invalid_scalar_classes(
    case: tuple[str, Any],
) -> None:
    field, invalid_value = case
    arguments: dict[str, Any] = {
        "n": _NN_3,
        "rank": _NN_0,
        "world_size": _POS_1,
    }
    arguments[field] = invalid_value
    with pytest.raises(BeartypeCallHintParamViolation):
        dump_module.iter_local_indices(**arguments)


def test_evaluate_completion_extracts_categorical_answer_letter() -> None:
    class _Task:
        eval_type = "categorical"

        def evaluate(self, conversation, predicted_letter):
            return predicted_letter == conversation["answer"]

    conversation = {"letters": ["A", "B"], "answer": "B"}

    outcome, error, predicted = dump_module.evaluate_completion(
        _Task(),
        conversation,
        "I choose B.",
    )

    assert outcome is True
    assert error is None
    assert predicted == "B"


def test_gather_rows_requires_complete_ddp_coverage() -> None:
    rows = [
        dump_module.DumpRow(
            index=0,
            user_message="u0",
            expected_answer="a0",
            completion="c0",
            outcome=True,
            eval_error=None,
            predicted_letter=None,
        ),
    ]
    with pytest.raises(AssertionError, match="dump coverage mismatch"):
        dump_module.gather_rows(
            rows,
            n_problems=_NN_2,
            ddp=False,
            ddp_world_size=_POS_1,
            all_gather_object=None,
        )


def test_dump_chat_responses_rejects_empty_feature_configs_before_loading_model() -> None:
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": "unused",
            "step": 1,
            "feature_configs": [],
        }
    )

    with pytest.raises(ValueError, match="requires feature_configs"):
        dump_module.dump_chat_responses(
            cfg,
            ddp=False,
            ddp_rank=_NN_0,
            ddp_world_size=_POS_1,
            device="cpu",
        )


def test_dump_chat_responses_rejects_empty_target_delta_configs() -> None:
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": "unused",
            "step": 1,
            "feature_configs": [{"name": "context_only", "features": {}}],
        }
    )

    with pytest.raises(ValueError, match="non-empty target/delta features"):
        dump_module.dump_chat_responses(
            cfg,
            ddp=False,
            ddp_rank=_NN_0,
            ddp_world_size=_POS_1,
            device="cpu",
        )


def test_dump_chat_responses_rejects_turn_checkpoint_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": "/checkpoints/turn",
            "step": 1,
            "feature_configs": [{"name": "delta", "features": {3: 1}}],
            "n_problems": 0,
        }
    )
    model = object.__new__(dump_module.EncoderDecoderGPT)
    monkeypatch.setattr(
        dump_module,
        "build_encdec_model",
        lambda *_args, **_kwargs: (
            model,
            object(),
            {"sft_segment_mode": "turn_delimited_fixed_tokens"},
        ),
    )
    monkeypatch.setattr(
        dump_module,
        "build_shared_context_feature_provider",
        lambda *_args, **_kwargs: pytest.fail("context provider reached"),
    )

    with pytest.raises(
        ValueError,
        match=r"dump_chat_responses.*turn_delimited_fixed_tokens",
    ):
        dump_module.dump_chat_responses(
            cfg,
            ddp=False,
            ddp_rank=_NN_0,
            ddp_world_size=_POS_1,
            device="cpu",
        )


@pytest.mark.parametrize("invalid_n_problems", [-1, True, 1.5])
def test_dump_chat_responses_rejects_invalid_problem_count_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    invalid_n_problems: object,
) -> None:
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": "unused",
            "step": 1,
            "feature_configs": [{"name": "delta", "features": {3: 1}}],
            "n_problems": invalid_n_problems,
        }
    )
    monkeypatch.setattr(
        dump_module,
        "build_encdec_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model load must not run")
        ),
    )

    with pytest.raises(TypeError):
        dump_module.dump_chat_responses(
            cfg,
            ddp=False,
            ddp_rank=_NN_0,
            ddp_world_size=_POS_1,
            device="cpu",
        )


def test_dump_one_feature_config_uses_prompt_context_provider(monkeypatch) -> None:
    calls: list[dict] = []

    class _Task:
        eval_type = "generative"

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            assert idx == 0
            return {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            }

        def evaluate(self, conversation, completion):
            return completion == "OK"

    class _Tokenizer:
        def render_for_completion(self, conversation):
            assert conversation["messages"][0]["content"] == "Q"
            return [10, 11, 12]

        def encode_special(self, token):
            assert token == "<assistant_end>"
            return 99

        def decode(self, token_ids):
            if list(token_ids) == [10, 11, 12]:
                return "PROMPT"
            assert list(token_ids) == [70, 71]
            return "OK"

    class _Wrapper:
        def conditioning_context(self, *, context_token_ids):
            calls.append({"context_token_ids": context_token_ids})
            return nullcontext()

    class _Engine:
        def __init__(self, wrapper, tokenizer):
            assert isinstance(wrapper, _Wrapper)
            assert isinstance(tokenizer, _Tokenizer)

        def generate_batch(self, prompt, *, num_samples, max_tokens, temperature, top_k):
            assert prompt == [10, 11, 12]
            assert num_samples == 1
            assert max_tokens == 8
            assert temperature == 0.0
            assert top_k == 50
            return [[10, 11, 12, 70, 71]], [[0, 0, 0, 1, 1]]

    context_provider = object()

    def _build_wrapper_for_features(model, tokenizer, *, features, context_provider, target_mode):
        calls.append(
            {
                "features": features,
                "context_provider": context_provider,
                "target_mode": target_mode,
            }
        )
        return _Wrapper()

    monkeypatch.setattr(dump_module, "build_task", lambda name: _Task())
    monkeypatch.setattr(dump_module, "build_wrapper_for_features", _build_wrapper_for_features)
    monkeypatch.setattr(dump_module, "Engine", _Engine)

    result = dump_module.dump_one_feature_config(
        feature_config=dump_module.FeatureConfig(name="delta", features={3: 1}),
        model=object.__new__(dump_module.EncoderDecoderGPT),
        tokenizer=_Tokenizer(),
        context_provider=context_provider,
        target_mode=TargetConditioningMode.NEXT,
        task_names=["GSM8K"],
        n_problems=_NN_1,
        max_new_tokens=8,
        temperature=0.0,
        top_k=50,
        autocast_ctx=nullcontext(),
        ddp=False,
        ddp_rank=_NN_0,
        ddp_world_size=_POS_1,
        all_gather_object=None,
    )

    assert result["GSM8K"][0].completion == "OK"
    assert result["GSM8K"][0].outcome is True
    assert calls == [
        {
            "features": {3: 1},
            "context_provider": context_provider,
            "target_mode": TargetConditioningMode.NEXT,
        },
        {"context_token_ids": [[10, 11, 12]]},
    ]
