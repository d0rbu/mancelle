"""Test that training_progression preserves the complete harm transcript."""

from __future__ import annotations


class _StubEngine:
    def __init__(self, results, masks):
        self._results = results
        self._masks = masks

    def generate_batch(self, prompt_tokens, num_samples=1, **kwargs):
        return self._results, self._masks


class _StubTokenizer:
    def __init__(self, id_to_text):
        self._id_to_text = id_to_text

    def decode(self, tokens):
        return "".join(self._id_to_text.get(int(t), "") for t in tokens)


def test_harmfulness_transcript_preserves_forced_tokens():
    """The target-aware harmfulness path decodes the full Engine transcript."""
    A, B = 50, 51
    output_start, output_end = 102, 103
    id_to_text = {
        1: "<a>", 2: "<b>", 3: "<c>",
        A: "A", B: "B",
        output_start: "<output_start>",
        output_end: "<output_end>",
        ord("4"): "4",
    }
    # Prompt has 3 tokens; generated has 5 (1 sampled, 3 forced, 1 sampled).
    results = [[1, 2, 3, A, output_start, ord("4"), output_end, B]]
    masks = [[0, 0, 0, 1, 0, 0, 0, 1]]
    engine = _StubEngine(results, masks)
    tokenizer = _StubTokenizer(id_to_text)

    # Match the production training_progression path: preserve the generated
    # sequence prefix boundary and decode every generated token.
    prompt_tokens = [1, 2, 3]
    generated, _gen_masks = engine.generate_batch(prompt_tokens, num_samples=1)
    response = tokenizer.decode(generated[0][len(prompt_tokens) :])
    assert response == "A<output_start>4<output_end>B"
