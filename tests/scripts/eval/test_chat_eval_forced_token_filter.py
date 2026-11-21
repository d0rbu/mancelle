"""Test that chat evaluation preserves the complete Engine transcript."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.eval.chat_eval import run_generative_eval


class _StubEngine:
    """Stub nanochat Engine that returns a fixed (results, masks) tuple
    for any generate_batch call.

    Each `generate_batch` call expects the prompt tokens as positional arg
    and returns ``(results, masks)`` with shape ``(num_samples, T_full)``.
    """

    def __init__(self, results, masks):
        self._results = results
        self._masks = masks
        self.calls = []

    def generate_batch(self, prompt_tokens, num_samples=1, **kwargs):
        self.calls.append((prompt_tokens, num_samples, kwargs))
        return self._results, self._masks


class _StubTokenizer:
    def __init__(self, id_to_text):
        self._id_to_text = id_to_text

    def decode(self, tokens):
        return "".join(self._id_to_text.get(int(t), "") for t in tokens)

    def render_for_completion(self, conv):
        return list(conv["messages"][-1].get("encoded_prompt", []))

    def encode_special(self, token):
        assert token == "<assistant_end>"
        return 13

    def get_bos_token_id(self):
        return 0


class _StubTask:
    """A length-1 task whose evaluate captures the completion string passed in."""

    eval_type = "generative"

    def __init__(self):
        self.captured_completion = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Pre-encoded prompt: tokens 1, 2, 3.
        return {
            "messages": [
                {"role": "user", "content": "ignored"},
                {"role": "assistant", "content": "", "encoded_prompt": [1, 2, 3]},
            ]
        }

    def evaluate(self, conversation, completion):
        self.captured_completion = completion
        return True


def test_run_generative_eval_preserves_forced_tokens_in_completion():
    """Forced tool-output tokens remain part of the scored transcript."""
    # prompt tokens (length 3) + 6 generated tokens (sampled + forced mix).
    # tokens: prompt + [A, output_start, "4", output_end, B, assistant_end]
    # masks:  [0,0,0,    1,     0,            0,     0,           1,  1]
    A = 50
    B = 51
    output_start, output_end = 102, 103
    assistant_end = 13
    bos = 0

    id_to_text = {
        1: "<a>", 2: "<b>", 3: "<c>",
        A: "A", B: "B",
        output_start: "<output_start>",
        output_end: "<output_end>",
        ord("4"): "4",
        assistant_end: "",
        bos: "",
    }

    # The engine's results include prompt + generated; masks are returned for
    # conditioning but do not alter the transcript text.
    results = [[1, 2, 3, A, output_start, ord("4"), output_end, B]]
    masks = [[0, 0, 0, 1, 0, 0, 0, 1]]
    engine = _StubEngine(results, masks)
    tokenizer = _StubTokenizer(id_to_text)
    task = _StubTask()

    # Drive run_generative_eval. Patch get_dist_info to return single-rank.
    with patch("scripts.eval.chat_eval.get_dist_info") as get_dist:
        get_dist.return_value = (False, 0, 0, 1)
        # model.get_device must exist for the function body.
        model = MagicMock()
        model.get_device.return_value = "cpu"
        # Disable target-aware encdec gating.
        del model.is_target_aware_encdec
        accuracy = run_generative_eval(
            task_object=task,
            tokenizer=tokenizer,
            model=model,
            engine=engine,
            num_samples=1,
            max_new_tokens=8,
            temperature=0.0,
            top_k=None,
            max_problems=1,
        )

    # The completion passed to evaluate includes the forced tool output.
    assert task.captured_completion == "A<output_start>4<output_end>B"
    # And accuracy should be 1.0 since _StubTask.evaluate returns True.
    assert accuracy == 1.0
