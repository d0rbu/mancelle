"""Tests for chat eval conditional encoder-decoder support."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch as t
from omegaconf import OmegaConf

import scripts.eval.chat_eval as chat_eval_module
from nanochat.engine import Engine
from scripts.eval.chat_eval import (
    prepare_model_for_chat_eval,
    run_categorical_eval,
    run_multi_config_chat_eval,
)
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.eval.dclm_eval import FeatureConfig
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    MultiTargetSegments,
    SelfConditionedTarget,
    SingleTargetSegment,
)
from steering.segmentation import token_to_char_offsets
from tests.conftest import MockTokenizer, _make_encdec, _make_gpt

MOCK_TASK_ACCURACY = 0.5


class RecordingTargetAwareModel:
    is_target_aware_encdec = True

    def __init__(self) -> None:
        self.context_calls = []

    def get_device(self):
        return t.device("cpu")

    def conditioning_context(self, target_mask=None, context_token_ids=None):
        model = self

        class _Ctx:
            def __enter__(self_inner):
                assert target_mask is not None
                model.context_calls.append((target_mask.clone(), context_token_ids))
                return model

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()

    def __call__(self, input_ids):
        logits = t.zeros(input_ids.size(0), input_ids.size(1), 128)
        # Make answer "A" win wherever the categorical eval reads logits.
        logits[:, :, 65] = 10.0
        return logits


class TinyCategoricalTask:
    eval_type = "categorical"

    def __init__(self) -> None:
        self.items = [
            {"tokens": [1, 2, 65], "letters": ["A", "B"], "answer": "A"},
            {"tokens": [3, 4, 5, 65], "letters": ["A", "B"], "answer": "A"},
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def evaluate(self, conversation, predicted_letter):
        return predicted_letter == conversation["answer"]


class TinyChatTokenizer(MockTokenizer):
    def render_for_completion(self, conversation):
        return list(conversation["tokens"])

    def encode(self, text, *args, **kwargs):
        if text == "A":
            return [65]
        if text == "B":
            return [66]
        return super().encode(text, *args, **kwargs)


class _SyntheticContextProvider:
    """Return one full-prefix feature segment for any prompt token sequence."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def get(self, token_ids):
        token_ids = [int(tok) for tok in token_ids]
        if not token_ids:
            return None
        char_end = token_to_char_offsets(self._tokenizer, token_ids)[-1]
        if char_end <= 0:
            return None
        return (
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(1,),
                char_start=0,
                char_end=int(char_end),
            ),
        )


def _wrap_encdec(model, tokenizer, features=None):
    return TargetAwareEncoderDecoderEvalWrapper(
        model,
        tokenizer,
        target=SingleTargetSegment(features=dict(features or {})),
        context_provider=_SyntheticContextProvider(tokenizer),
    )


class TestMockTokenizerRenderForCompletion:
    """MockTokenizer must support render_for_completion for chat eval tests."""

    def test_render_for_completion_returns_token_ids(self) -> None:
        tok = MockTokenizer()
        conversation = {"text": "hello"}
        result = tok.render_for_completion(conversation)
        assert isinstance(result, list)
        assert all(isinstance(tok_id, int) for tok_id in result)
        assert len(result) > 0


def test_categorical_eval_passes_target_aware_conditioning_context():
    model = RecordingTargetAwareModel()
    tokenizer = TinyChatTokenizer()

    acc = run_categorical_eval(
        TinyCategoricalTask(),
        tokenizer,
        model,
        batch_size=2,
        max_problems=2,
    )

    assert acc == 1.0
    assert len(model.context_calls) == 1
    target_mask, context_token_ids = model.context_calls[0]
    assert target_mask.tolist() == [
        [False, False, True, False],
        [False, False, False, True],
    ]
    assert context_token_ids == [[1, 2, 65], [3, 4, 5, 65]]


def test_categorical_eval_with_real_target_aware_wrapper():
    """Exercise categorical eval through a real full-context encdec wrapper."""
    encdec = _make_encdec()
    tok = MockTokenizer()
    wrapped = _wrap_encdec(encdec, tok, features={0: 1})

    # Tiny categorical task whose tokens are valid for the real tokenizer.
    class _RealTokTask:
        eval_type = "categorical"

        def __init__(self) -> None:
            self.items = [
                {"tokens": [1, 2, 3], "letters": ["A", "B"], "answer": "A"},
                {"tokens": [4, 5, 6, 7], "letters": ["A", "B"], "answer": "B"},
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

        def evaluate(self, conversation, predicted_letter):
            return predicted_letter == conversation["answer"]

    class _RealTokWithRender(MockTokenizer):
        def render_for_completion(self, conversation):
            return list(conversation["tokens"])

    real_tok = _RealTokWithRender()

    # Simply assert the eval path runs without error and returns a float
    # in [0, 1]. Concrete accuracy is undefined for an untrained tiny
    # encdec; correctness here is "the categorical path executes through
    # the real wrapper's conditioning_context + forward".
    with t.no_grad():
        acc = run_categorical_eval(
            _RealTokTask(),
            real_tok,
            wrapped,
            batch_size=2,
            max_problems=2,
        )
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


class TestPrepareModelForChatEval:
    """Test prepare_model_for_chat_eval dispatch logic."""

    def test_decoder_only_returns_model_unchanged(self) -> None:
        model = _make_gpt()
        result = prepare_model_for_chat_eval(
            model, SingleTargetSegment(features={})
        )
        assert result is model

    def test_decoder_only_with_empty_features_returns_model_unchanged(self) -> None:
        model = _make_gpt()
        result = prepare_model_for_chat_eval(
            model, SingleTargetSegment(features={})
        )
        assert result is model

    def test_decoder_only_rejects_features(self) -> None:
        model = _make_gpt()
        with pytest.raises(ValueError, match=r"[Dd]ecoder.only"):
            prepare_model_for_chat_eval(
                model, SingleTargetSegment(features={0: 1, 5: 2})
            )

    def test_decoder_only_rejects_self_conditioning_marker(self) -> None:
        model = _make_gpt()
        with pytest.raises(ValueError, match=r"[Dd]ecoder.only"):
            prepare_model_for_chat_eval(model, SelfConditionedTarget())

    def test_encdec_rejects_context_only_standalone_config(self) -> None:
        model = _make_encdec()
        with pytest.raises(ValueError, match="target_real"):
            prepare_model_for_chat_eval(
                model, SingleTargetSegment(features={})
            )

    def test_encdec_rejects_target_only_standalone_config(self) -> None:
        model = _make_encdec()
        with pytest.raises(ValueError, match="target-only"):
            prepare_model_for_chat_eval(
                model,
                SingleTargetSegment(features={0: 1, 5: 2}),
            )

    def test_multitarget_standalone_encdec_fails_loudly(self) -> None:
        model = _make_encdec()
        with pytest.raises(ValueError, match="target_real"):
            prepare_model_for_chat_eval(
                model,
                MultiTargetSegments(
                    segments=[{0: 1}, {1: 2}],
                    token_assignments=t.tensor([0, 1], dtype=t.long),
                ),
            )

    def test_encdec_wrapper_has_config_and_device(self) -> None:
        model = _make_encdec()
        tok = MockTokenizer()
        result = _wrap_encdec(model, tok, features={0: 1})
        # Engine reads model.config.n_kv_head, n_embd, n_head, n_layer
        assert hasattr(result, "config")
        assert hasattr(result.config, "n_kv_head")
        assert hasattr(result.config, "n_embd")
        assert hasattr(result, "get_device")


class TestWrapperEvalCompatibility:
    """Verify TargetAwareEncoderDecoderEvalWrapper works with both eval paths."""

    def test_categorical_forward_returns_logits(self) -> None:
        """Categorical eval calls model(prompt_ids) and expects (B, T, V) logits."""
        model = _make_encdec()
        tok = MockTokenizer()
        wrapped = _wrap_encdec(model, tok, features={0: 1})

        B, T = 2, 16
        prompt_ids = t.randint(0, 64, (B, T))  # vocab_size=128 in tiny config
        target_mask = t.zeros(B, T, dtype=t.bool)
        target_mask[:, -1] = True
        context_token_ids = [prompt_ids[i, :-1].tolist() for i in range(B)]
        with t.no_grad():
            with wrapped.conditioning_context(
                target_mask=target_mask,
                context_token_ids=context_token_ids,
            ):
                logits = wrapped(prompt_ids)

        assert logits.shape == (B, T, 128)  # (B, T, vocab_size)
        assert logits.dtype in (t.float32, t.bfloat16)

    def test_categorical_forward_with_features(self) -> None:
        """Same test but with an explicit target feature delta."""
        model = _make_encdec()
        tok = MockTokenizer()
        wrapped = _wrap_encdec(model, tok, features={0: 1, 5: 2})

        B, T = 2, 16
        assistant_start_id = tok.encode_special("<assistant_start>")
        prompt_ids = t.randint(0, 64, (B, T))
        prompt_ids[:, -1] = assistant_start_id
        target_mask = t.zeros(B, T, dtype=t.bool)
        target_mask[:, -1] = True
        context_token_ids = [prompt_ids[i].tolist() for i in range(B)]
        with t.no_grad():
            with wrapped.conditioning_context(
                target_mask=target_mask,
                context_token_ids=context_token_ids,
            ):
                logits = wrapped(prompt_ids)

        assert logits.shape == (B, T, 128)

    def test_decoder_only_categorical_forward_unchanged(self) -> None:
        """Decoder-only model still works with the same calling convention."""
        model = _make_gpt()
        B, T = 2, 16
        prompt_ids = t.randint(0, 64, (B, T))
        with t.no_grad():
            logits = model(prompt_ids)

        assert logits.shape == (B, T, 128)

    def test_engine_generate_with_wrapper(self) -> None:
        """Engine.generate_batch must work with TargetAwareEncoderDecoderEvalWrapper."""
        model = _make_encdec()
        tok = MockTokenizer()
        wrapped = _wrap_encdec(model, tok, features={0: 1})
        engine = Engine(wrapped, tok)

        prompt_tokens = tok.encode("hello", prepend=tok.get_bos_token_id())
        results, masks = engine.generate_batch(
            prompt_tokens,
            num_samples=1,
            max_tokens=4,
            temperature=0.0,
        )

        assert len(results) == 1
        assert len(results[0]) >= len(prompt_tokens)  # at least the prompt
        assert all(isinstance(tok_id, int) for tok_id in results[0])

    def test_engine_generate_with_features(self) -> None:
        """Engine.generate_batch with full-context conditioning."""
        model = _make_encdec()
        tok = MockTokenizer()
        wrapped = _wrap_encdec(model, tok, features={0: 1, 5: 2})
        engine = Engine(wrapped, tok)

        conversation = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]}
        prompt_tokens = tok.render_for_completion(conversation)
        with wrapped.conditioning_context(context_token_ids=[prompt_tokens]):
            results, masks = engine.generate_batch(
                prompt_tokens,
                num_samples=1,
                max_tokens=4,
                temperature=0.0,
            )

        assert len(results) == 1
        assert len(results[0]) >= len(prompt_tokens)


class TestRunMultiConfigChatEval:
    """Test run_multi_config_chat_eval orchestration."""

    @patch("scripts.eval.chat_eval.run_chat_eval", return_value=MOCK_TASK_ACCURACY)
    def test_single_decoder_only_config(self, mock_run) -> None:
        model = _make_gpt()
        tok = MockTokenizer()
        configs = [FeatureConfig(name="baseline", features={})]

        results = run_multi_config_chat_eval(
            model,
            tok,
            configs,
            task_names=["ARC-Easy"],
            batch_size=1,
            num_samples=1,
            max_new_tokens=32,
            temperature=0.0,
            top_k=50,
            max_problems=1,
        )

        assert len(results) == 1
        assert results[0]["name"] == "baseline"
        assert "ARC-Easy" in results[0]["results"]
        mock_run.assert_called_once()

    @patch("scripts.eval.chat_eval.run_chat_eval", return_value=MOCK_TASK_ACCURACY)
    def test_multi_config_encdec_rejects_standalone_configs(self, mock_run) -> None:
        model = _make_encdec()
        tok = MockTokenizer()
        configs = [
            FeatureConfig(name="context_only", features={}),
            FeatureConfig(name="topk", features={0: 1, 5: 2}),
        ]

        with pytest.raises(ValueError, match="target_real"):
            run_multi_config_chat_eval(
                model,
                tok,
                configs,
                task_names=["ARC-Easy", "MMLU"],
                batch_size=1,
                num_samples=1,
                max_new_tokens=32,
                temperature=0.0,
                top_k=50,
                max_problems=1,
            )
        mock_run.assert_not_called()

    @patch("scripts.eval.chat_eval.run_chat_eval", return_value=MOCK_TASK_ACCURACY)
    def test_decoder_only_rejects_features_in_multi_config(self, mock_run) -> None:
        model = _make_gpt()
        tok = MockTokenizer()
        configs = [FeatureConfig(name="bad", features={0: 1})]

        with pytest.raises(ValueError, match=r"[Dd]ecoder.only"):
            run_multi_config_chat_eval(
                model,
                tok,
                configs,
                task_names=["ARC-Easy"],
                batch_size=1,
                num_samples=1,
                max_new_tokens=32,
                temperature=0.0,
                top_k=50,
                max_problems=1,
            )

    @patch("scripts.eval.chat_eval.run_chat_eval", return_value=MOCK_TASK_ACCURACY)
    def test_chatcore_metric_computed(self, mock_run) -> None:
        model = _make_gpt()
        tok = MockTokenizer()
        all_tasks = [
            "ARC-Easy",
            "ARC-Challenge",
            "MMLU",
            "GSM8K",
            "HumanEval",
            "SpellingBee",
        ]
        configs = [FeatureConfig(name="baseline", features={})]

        results = run_multi_config_chat_eval(
            model,
            tok,
            configs,
            task_names=all_tasks,
            batch_size=1,
            num_samples=1,
            max_new_tokens=32,
            temperature=0.0,
            top_k=50,
            max_problems=1,
        )

        assert "chatcore_metric" in results[0]
        assert isinstance(results[0]["chatcore_metric"], float)
        # Verify actual formula: centered_acc = (acc - baseline) / (1 - baseline)
        # ARC-Easy/Challenge/MMLU: (0.5 - 0.25) / (1 - 0.25) = 1/3
        # GSM8K/HumanEval/SpellingBee: (0.5 - 0.0) / (1 - 0.0) = 0.5
        expected = (1 / 3 * 3 + 0.5 * 3) / 6
        assert results[0]["chatcore_metric"] == pytest.approx(expected)


class _InactiveBucketSentinel(Exception):
    """Raised by patched inactive-bucket validators in entrypoint wiring tests."""


def test_main_calls_inactive_bucket_guard_for_encdec(tmp_path, monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "device_type": "cpu",
            "dtype": "float32",
            "source": "encdec",
            "model_tag": "d12",
            "step": None,
            "task_name": "ARC-Easy",
            "feature_configs": "ignored",
        }
    )
    model = _make_encdec()
    meta = {"step": 0, "user_config": {}}
    cfg.step = 0
    (tmp_path / "meta_000000.json").write_text(json.dumps(meta))
    monkeypatch.setattr(chat_eval_module, "get_checkpoint_dir", lambda *args, **kwargs: str(tmp_path))

    def _boom(configs, *, num_buckets, checkpoint_meta, checkpoint_label="checkpoint"):
        del configs, checkpoint_label
        assert num_buckets == model.encoder.config.num_buckets
        assert checkpoint_meta is meta
        raise _InactiveBucketSentinel("guard called")

    with (
        patch.object(chat_eval_module, "compute_init", return_value=(False, 0, 0, 1, "cpu")),
        patch.object(
            chat_eval_module,
            "load_model",
            return_value=(model, MockTokenizer(), meta),
        ),
        patch.object(
            chat_eval_module,
            "parse_feature_configs",
            return_value=[
                FeatureConfig(
                    name="mixed",
                    features={0: model.encoder.config.num_buckets - 1},
                )
            ],
        ),
        patch.object(
            chat_eval_module,
            "validate_inactive_bucket_feature_configs",
            side_effect=_boom,
        ),
        pytest.raises(_InactiveBucketSentinel, match="guard called"),
    ):
        chat_eval_module.main.__wrapped__(cfg)
