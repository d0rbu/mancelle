"""Byte-identity parity tests: batched_generate_seeds output must match
independent single-seed Hugging Face generation.

These tests are the gate for PR-2 merge. They use a real CPU-resident
HF model (sshleifer/tiny-gpt2) so the random draws inside HF generate AND
inside multinomial both fire and we can compare token streams.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.eval.axbench_base.batched_generation import (
    batched_generate_seeds,
    cross_prefix_batched_generate_seeds,
)


@pytest.fixture
def real_tiny_hf_model():
    """Load a real tiny HF causal LM for parity testing.

    Uses the gpt2-small or sshleifer/tiny-gpt2 architecture so the test
    runs on CPU without GPU. Skips if transformers is not installed
    or if the model cannot be downloaded (e.g., no network in CI).
    """
    transformers = pytest.importorskip("transformers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer

    model_id = "sshleifer/tiny-gpt2"
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as exc:
        pytest.fail(
            f"Parity-test fixture {model_id!r} unavailable: {exc}. "
            "This is the byte-identity gate for PR-2 — it MUST run, not skip. "
            "Ensure HF cache is populated or network is available in CI."
        )

    model.eval()
    return model, tok


def _make_base_model_wrapper(hf_model):
    """Wrap an HF model in the shape `_hf_generation_model` expects."""

    class _Cfg:
        n_ctx = 256
        n_layers = 2

    class _NnterpWrapper:
        def __init__(self, m):
            self._model = m

    class _BaseModel:
        cfg = _Cfg()
        nnterp_model = _NnterpWrapper(hf_model)

        def parameters(self):
            return hf_model.parameters()

    return _BaseModel()


def _per_prefix_batched_outputs(
    base_model,
    prompts: list[list[int]],
    seed_rows: list[list[int]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    stop_token_ids: set[int],
) -> list[list[list[int]]]:
    return [
        batched_generate_seeds(
            base_model=base_model,
            token_ids=prompt,
            seeds=seeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_token_ids=stop_token_ids,
        )
        for prompt, seeds in zip(prompts, seed_rows, strict=True)
    ]


@pytest.mark.parametrize(
    "seeds",
    [
        [7],
        [7, 11],
        [7, 11, 13],
        [7, 11, 13, 17, 19],
        [42, 100, 200, 300, 400],
        [1, 2, 3, 4, 5],
        [10000, 20000, 30000, 40000, 50000],
        [123456, 234567, 345678, 456789, 567890],
        [999, 1000, 1001, 1002, 1003],
        [98765, 87654, 76543, 65432, 54321],
    ],
)
def test_batched_vs_sequential_token_identity(real_tiny_hf_model, seeds):
    """For each set of seeds: sequential per-seed generate produces the
    same token streams as the batched kernel."""
    hf_model, tok = real_tiny_hf_model
    base_model = _make_base_model_wrapper(hf_model)

    prompt = tok.encode("hello world")
    max_new_tokens = 16
    temperature = 0.7
    top_k = 50
    stop_token_ids = (
        {tok.eos_token_id} if tok.eos_token_id is not None else set()
    )

    # Sequential reference: invoke Hugging Face generation one seed at a time.
    sequential_outputs: list[list[int]] = []
    for s in seeds:
        with t.random.fork_rng():
            t.manual_seed(s)
            input_ids = t.tensor([prompt], dtype=t.long)
            attention = t.ones_like(input_ids)
            generated = hf_model.generate(
                input_ids,
                attention_mask=attention,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                eos_token_id=sorted(stop_token_ids) if stop_token_ids else None,
            )
        cont = generated[0, len(prompt):].tolist()
        # Trim trailing stop tokens to match the batched-kernel contract.
        trimmed: list[int] = []
        for tid in cont:
            if tid in stop_token_ids:
                break
            trimmed.append(int(tid))
        sequential_outputs.append(trimmed)

    # Batched kernel
    batched_outputs = batched_generate_seeds(
        base_model=base_model,
        token_ids=prompt,
        seeds=seeds,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=stop_token_ids,
    )

    assert batched_outputs == sequential_outputs, (
        f"Byte-identity violated for seeds {seeds}:\n"
        f"  sequential: {sequential_outputs}\n"
        f"  batched:    {batched_outputs}"
    )


def test_batched_vs_sequential_under_top_k_constraint(real_tiny_hf_model):
    """Same parity but with top_k=10 (where ties are common)."""
    hf_model, tok = real_tiny_hf_model
    base_model = _make_base_model_wrapper(hf_model)

    prompt = tok.encode("the quick brown fox")
    seeds = [7, 11, 13]
    sequential_outputs: list[list[int]] = []
    for s in seeds:
        with t.random.fork_rng():
            t.manual_seed(s)
            input_ids = t.tensor([prompt], dtype=t.long)
            attention = t.ones_like(input_ids)
            generated = hf_model.generate(
                input_ids,
                attention_mask=attention,
                max_new_tokens=8,
                do_sample=True,
                temperature=0.7,
                top_k=10,
                use_cache=True,
            )
        sequential_outputs.append(generated[0, len(prompt):].tolist())

    batched_outputs = batched_generate_seeds(
        base_model=base_model,
        token_ids=prompt,
        seeds=seeds,
        max_new_tokens=8,
        temperature=0.7,
        top_k=10,
        stop_token_ids=set(),
    )
    assert batched_outputs == sequential_outputs


def test_batched_vs_sequential_with_stop_token_truncation(real_tiny_hf_model):
    """Same parity with stop tokens injected mid-generation."""
    hf_model, tok = real_tiny_hf_model
    base_model = _make_base_model_wrapper(hf_model)

    prompt = tok.encode("once upon a time")
    seeds = [42, 43, 44]
    # Pick a few common-ish token ids as stop tokens to force truncation
    stop_ids = {tok.encode(" the")[0] if tok.encode(" the") else 1000}

    sequential_outputs: list[list[int]] = []
    for s in seeds:
        with t.random.fork_rng():
            t.manual_seed(s)
            input_ids = t.tensor([prompt], dtype=t.long)
            attention = t.ones_like(input_ids)
            generated = hf_model.generate(
                input_ids,
                attention_mask=attention,
                max_new_tokens=12,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                use_cache=True,
                eos_token_id=sorted(stop_ids),
            )
        cont = generated[0, len(prompt):].tolist()
        trimmed: list[int] = []
        for tid in cont:
            if int(tid) in stop_ids:
                break
            trimmed.append(int(tid))
        sequential_outputs.append(trimmed)

    batched_outputs = batched_generate_seeds(
        base_model=base_model,
        token_ids=prompt,
        seeds=seeds,
        max_new_tokens=12,
        temperature=0.7,
        top_k=50,
        stop_token_ids=stop_ids,
    )
    assert batched_outputs == sequential_outputs


def test_cross_prefix_batched_matches_per_prefix_batched_tiny_gpt2(
    real_tiny_hf_model,
):
    """Variable-length cross-prefix batching must match serial PR-2 calls."""
    hf_model, tok = real_tiny_hf_model
    base_model = _make_base_model_wrapper(hf_model)

    prompts = [
        tok.encode("short"),
        tok.encode("a medium length prompt"),
        tok.encode("a much longer prefix that forces left padding"),
    ]
    seed_rows = [
        [7, 11],
        [13, 17],
        [19, 23],
    ]
    max_new_tokens = 8
    temperature = 0.7
    top_k = 50
    stop_token_ids = (
        {tok.eos_token_id} if tok.eos_token_id is not None else set()
    )

    expected = [
        batched_generate_seeds(
            base_model=base_model,
            token_ids=prompt,
            seeds=seeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_token_ids=stop_token_ids,
        )
        for prompt, seeds in zip(prompts, seed_rows, strict=True)
    ]
    actual = cross_prefix_batched_generate_seeds(
        base_model=base_model,
        token_id_rows=prompts,
        seed_rows=seed_rows,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=stop_token_ids,
        prefill_mode="repeated",
    )

    assert actual == expected


def test_cross_prefix_batched_matches_per_prefix_batched_tiny_gpt2_greedy_left_pad(
    real_tiny_hf_model,
):
    """Greedy rows with uneven prompt lengths must match singleton prompts."""
    hf_model, tok = real_tiny_hf_model
    base_model = _make_base_model_wrapper(hf_model)

    prompts = [
        tok.encode("short"),
        tok.encode("a prefix with enough words to require left padding"),
        tok.encode("mid length prompt"),
        tok.encode("x"),
    ]
    seed_rows = [[7], [11, 13], [17], [19, 23, 29]]
    max_new_tokens = 6
    stop_token_ids: set[int] = set()

    expected = _per_prefix_batched_outputs(
        base_model,
        prompts,
        seed_rows,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=None,
        stop_token_ids=stop_token_ids,
    )
    actual = cross_prefix_batched_generate_seeds(
        base_model=base_model,
        token_id_rows=prompts,
        seed_rows=seed_rows,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=None,
        stop_token_ids=stop_token_ids,
        prefill_mode="repeated",
    )

    assert actual == expected


def test_cross_prefix_batched_matches_per_prefix_batched_tiny_gpt2_uneven_seeds(
    real_tiny_hf_model,
):
    """Uneven seed counts must not perturb per-row RNG streams."""
    hf_model, tok = real_tiny_hf_model
    base_model = _make_base_model_wrapper(hf_model)

    prompts = [
        tok.encode("alpha"),
        tok.encode("beta beta beta"),
        tok.encode("gamma prefix with more context"),
        tok.encode("delta"),
    ]
    seed_rows = [[7], [11, 13, 17], [19, 23], [29, 31, 37, 41]]
    max_new_tokens = 7
    temperature = 0.7
    top_k = 50

    expected = _per_prefix_batched_outputs(
        base_model,
        prompts,
        seed_rows,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=set(),
    )
    actual = cross_prefix_batched_generate_seeds(
        base_model=base_model,
        token_id_rows=prompts,
        seed_rows=seed_rows,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=set(),
        prefill_mode="repeated",
    )

    assert actual == expected


def test_cross_prefix_batched_matches_per_prefix_batched_tiny_gpt2_stop_tokens(
    real_tiny_hf_model,
):
    """Stop-token truncation must match per-prefix generation row by row."""
    hf_model, tok = real_tiny_hf_model
    base_model = _make_base_model_wrapper(hf_model)

    prompts = [
        tok.encode("stop token prompt one"),
        tok.encode("stop token prompt two with more context"),
        tok.encode("stop token prompt three"),
    ]
    seed_rows = [[101, 103], [107, 109], [113, 127]]
    max_new_tokens = 8
    temperature = 0.7
    top_k = 50
    no_stop = _per_prefix_batched_outputs(
        base_model,
        prompts,
        seed_rows,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=set(),
    )
    first_generated_tokens = [
        continuation[0]
        for prefix_rows in no_stop
        for continuation in prefix_rows
        if continuation
    ]
    assert first_generated_tokens
    stop_token_ids = {first_generated_tokens[0]}

    expected = _per_prefix_batched_outputs(
        base_model,
        prompts,
        seed_rows,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=stop_token_ids,
    )
    actual = cross_prefix_batched_generate_seeds(
        base_model=base_model,
        token_id_rows=prompts,
        seed_rows=seed_rows,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=stop_token_ids,
        prefill_mode="repeated",
    )

    assert actual == expected
    assert any(
        len(continuation) < max_new_tokens
        for prefix_rows in actual
        for continuation in prefix_rows
    )
