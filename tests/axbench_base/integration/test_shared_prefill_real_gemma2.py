"""Very-slow production Gemma2 validation for Stage 0c prefill modes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering.eval.axbench_base.batched_generation import (
    cross_prefix_batched_generate_seeds,
)
from steering.eval.axbench_base.config import VerificationPrefillMode

_MODEL_ID = "google/gemma-2-2b"
_FROZEN_PROMPTS = [
    "Mathematical proofs often begin by fixing notation and stating the exact assumptions. ",
    "The coastal observatory recorded wind, pressure, and temperature before the storm arrived. ",
    "In the archive, each letter is catalogued with its date, author, recipient, and provenance. ",
    "A careful software review separates the public contract from implementation details. ",
]


def _base_model(hf_model: object) -> SimpleNamespace:
    return SimpleNamespace(
        nnterp_model=SimpleNamespace(
            _model=hf_model,
        )
    )


@pytest.mark.very_slow
def test_production_gemma2_prefill_modes_are_independently_exact_for_64_tokens() -> None:
    if not t.cuda.is_available():
        pytest.skip("production Gemma2 validation requires CUDA")
    try:
        tokenizer: Any = AutoTokenizer.from_pretrained(
            _MODEL_ID,
            local_files_only=True,
        )
        # Preserve the production probe's canonical load order: materialize
        # float32 weights on CPU, move to CUDA, then convert to bfloat16.
        model: Any = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            dtype=t.float32,
            local_files_only=True,
        )
    except OSError as exc:
        pytest.skip(f"cached {_MODEL_ID} snapshot unavailable: {exc}")

    model = model.to("cuda").to(dtype=t.bfloat16).eval()
    base_model = _base_model(model)
    encoded = [
        list(tokenizer.encode(prompt, add_special_tokens=True))
        for prompt in _FROZEN_PROMPTS
    ]
    cases = [
        (encoded[:1], [[710_001, 710_003, 710_009, 710_011, 710_017]]),
        (
            encoded,
            [
                [711_001, 711_003, 711_009, 711_011, 711_017],
                [712_001, 712_003, 712_009, 712_011, 712_017],
                [713_001, 713_003, 713_009, 713_011, 713_017],
                [714_001, 714_003, 714_009, 714_011, 714_017],
            ],
        ),
    ]

    try:
        for mode in VerificationPrefillMode:
            for token_rows, seed_rows in cases:
                kwargs = {
                    "seed_rows": seed_rows,
                    "max_new_tokens": 64,
                    "temperature": 0.7,
                    "top_k": 50,
                    "stop_token_ids": set(),
                    "prefill_mode": mode,
                }
                first = cross_prefix_batched_generate_seeds(
                    base_model,
                    token_rows,
                    **kwargs,
                )
                second = cross_prefix_batched_generate_seeds(
                    base_model,
                    token_rows,
                    **kwargs,
                )
                assert first == second
                assert all(
                    len(continuation) == 64
                    and all(isinstance(token, int) for token in continuation)
                    for prefix_rows in first
                    for continuation in prefix_rows
                )
    finally:
        del base_model, model
        t.cuda.empty_cache()
