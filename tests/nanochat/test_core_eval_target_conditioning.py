from __future__ import annotations

import torch

from nanochat.core_eval import (
    TaskMeta,
    TaskType,
    evaluate_example,
    forward_model,
)


class RecordingCoreModel:
    is_target_aware_encdec = True
    max_seq_len = None

    def __init__(self) -> None:
        self.context_calls = []

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
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), 32)
        if input_ids.size(1) > 4:
            # Multiple-choice tests score the answer token from position 3.
            logits[0, 3, int(input_ids[0, 4])] = 10.0
            if input_ids.size(0) > 1:
                logits[1, 3, int(input_ids[1, 4])] = -10.0
        return logits


class TinyCoreTokenizer:
    def get_bos_token_id(self):
        return 0

    def __call__(self, prompts, prepend=None):
        tokens = []
        for prompt in prompts:
            if prompt.endswith("A"):
                seq = [1, 2, 3, 10]
            elif prompt.endswith("B"):
                seq = [1, 2, 3, 11]
            else:
                seq = [1, 2, 3]
            tokens.append(([prepend] if prepend is not None else []) + seq)
        return tokens


def test_forward_model_passes_target_aware_conditioning_context():
    model = RecordingCoreModel()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    target_mask = torch.tensor([[False, True, False]])
    context_token_ids = [[1, 2]]

    losses, predictions = forward_model(
        model,
        input_ids,
        target_mask=target_mask,
        context_token_ids=context_token_ids,
    )

    assert losses.shape == input_ids.shape
    assert predictions.shape == input_ids.shape
    assert len(model.context_calls) == 1
    seen_mask, seen_context = model.context_calls[0]
    assert torch.equal(seen_mask, target_mask)
    assert seen_context == context_token_ids


def test_evaluate_example_builds_core_target_mask_and_context_tokens():
    model = RecordingCoreModel()
    tokenizer = TinyCoreTokenizer()
    data = [{"query": "Q", "choices": ["A", "B"], "gold": 0}]
    task_meta = TaskMeta(
        task_type=TaskType.MULTIPLE_CHOICE,
        num_fewshot=0,
        continuation_delimiter=" ",
    )

    assert evaluate_example(0, model, tokenizer, data, "cpu", task_meta)

    assert len(model.context_calls) == 1
    target_mask, context_token_ids = model.context_calls[0]
    assert target_mask.tolist() == [
        [False, False, False, True, False],
        [False, False, False, True, False],
    ]
    assert context_token_ids == [[0, 1, 2, 3], [0, 1, 2, 3]]


class RecordingLMModel(RecordingCoreModel):
    def __call__(self, input_ids):
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), 32)
        for row in range(input_ids.size(0)):
            for pos in range(input_ids.size(1) - 1):
                logits[row, pos, int(input_ids[row, pos + 1])] = 10.0
        return logits


class TinyLMTokenizer:
    def get_bos_token_id(self):
        return 0

    def __call__(self, prompts, prepend=None):
        tokens = []
        for prompt in prompts:
            if prompt == "ctx":
                seq = [1, 2, 3]
            elif prompt == "ctx XY":
                seq = [1, 2, 3, 4, 5]
            else:
                raise AssertionError(f"unexpected prompt: {prompt!r}")
            tokens.append(([prepend] if prepend is not None else []) + seq)
        return tokens


def test_evaluate_example_builds_lm_target_mask_one_position_before_continuation():
    model = RecordingLMModel()
    tokenizer = TinyLMTokenizer()
    data = [{"context": "ctx", "continuation": "XY"}]
    task_meta = TaskMeta(
        task_type=TaskType.LANGUAGE_MODELING,
        num_fewshot=0,
        continuation_delimiter=" ",
    )

    assert evaluate_example(0, model, tokenizer, data, "cpu", task_meta)

    assert len(model.context_calls) == 1
    target_mask, context_token_ids = model.context_calls[0]
    assert target_mask.tolist() == [[False, False, False, True, True, False]]
    assert context_token_ids == [[0, 1, 2, 3]]
