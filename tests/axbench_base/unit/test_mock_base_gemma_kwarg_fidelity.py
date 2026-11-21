"""Kwarg fidelity tests for the MockBaseGemma stub (PR-1).

PR-2's batched-generation parity tests must verify that the custom
batched-generation kernel propagates sampling kwargs (``do_sample``,
``temperature``, ``top_p``, ``pad_token_id``, ``eos_token_id``,
``attention_mask``, ``generation_config``, ...) to the underlying
``model.generate`` call. The mock must therefore (a) accept all
real-Gemma2 generate kwargs without raising and (b) record what it
received so tests can assert propagation.

Without these tests the mock could silently drop a kwarg that PR-2
relies on, masking a kernel-propagation bug.
"""

from __future__ import annotations

import pytest
import torch as t

from tests.axbench_base._mock_models import MockHFGenerateModel


@pytest.fixture
def mock_model() -> MockHFGenerateModel:
    return MockHFGenerateModel({0: [10, 11, 12, 13]})


@pytest.mark.parametrize(
    "kwarg_name, kwarg_value",
    [
        ("max_new_tokens", 4),
        ("do_sample", True),
        ("temperature", 0.7),
        ("top_p", 0.95),
        ("top_k", 50),
        ("pad_token_id", 0),
        ("eos_token_id", 1),
        ("num_return_sequences", 1),
        ("repetition_penalty", 1.1),
        ("output_scores", False),
        ("return_dict_in_generate", False),
        ("attention_mask", t.ones(1, 4, dtype=t.long)),
    ],
)
def test_generate_accepts_real_gemma2_kwargs(
    mock_model, kwarg_name, kwarg_value,
):
    """Each real-Gemma2 generate kwarg must be accepted without
    TypeError. ``**kwargs`` makes this trivially true today, but the
    test pins the contract so a future refactor that names individual
    kwargs cannot accidentally drop one."""
    t.manual_seed(0)
    input_ids = t.zeros((1, 4), dtype=t.long)
    out = mock_model.generate(input_ids, **{kwarg_name: kwarg_value})
    assert out.shape[0] == 1


def test_generate_records_kwargs_for_propagation_assertions(mock_model):
    """``_observed_kwargs`` records every kwargs dict received so PR-2
    tests can assert the kernel propagated each kwarg into the
    underlying ``model.generate`` call."""
    t.manual_seed(0)
    input_ids = t.zeros((1, 4), dtype=t.long)
    mock_model.generate(
        input_ids,
        max_new_tokens=2,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=0,
        eos_token_id=1,
    )
    assert len(mock_model._observed_kwargs) == 1
    seen = mock_model._observed_kwargs[0]
    assert seen["max_new_tokens"] == 2
    assert seen["do_sample"] is True
    assert seen["temperature"] == pytest.approx(0.7)
    assert seen["top_p"] == pytest.approx(0.95)
    assert seen["pad_token_id"] == 0
    assert seen["eos_token_id"] == 1


def test_generate_observed_kwargs_grows_per_call(mock_model):
    """Each ``generate()`` call appends a separate kwargs dict so PR-2
    parity tests can compare per-call behavior across the batched and
    sequential paths."""
    t.manual_seed(0)
    input_ids = t.zeros((1, 4), dtype=t.long)
    mock_model.generate(input_ids, max_new_tokens=1)
    mock_model.generate(input_ids, max_new_tokens=2, do_sample=False)
    mock_model.generate(input_ids, max_new_tokens=3, top_p=0.9)
    assert len(mock_model._observed_kwargs) == 3
    assert mock_model._observed_kwargs[0] == {"max_new_tokens": 1}
    assert mock_model._observed_kwargs[1] == {
        "max_new_tokens": 2, "do_sample": False,
    }
    assert mock_model._observed_kwargs[2] == {
        "max_new_tokens": 3, "top_p": 0.9,
    }


def test_generate_observed_kwargs_isolated_from_caller_mutation(mock_model):
    """``_observed_kwargs`` stores a copy of each kwargs dict so that
    if the caller mutates the dict after the call, the recorded value
    is unaffected (defensive against PR-2 batched-generation kernels
    that may reuse a single kwargs dict across iterations)."""
    t.manual_seed(0)
    input_ids = t.zeros((1, 4), dtype=t.long)
    shared_kwargs = {"max_new_tokens": 4, "do_sample": True}
    mock_model.generate(input_ids, **shared_kwargs)
    shared_kwargs["max_new_tokens"] = 999
    shared_kwargs["do_sample"] = False
    seen = mock_model._observed_kwargs[0]
    assert seen["max_new_tokens"] == 4, (
        "recorded kwargs must not reflect post-call mutation by caller"
    )
    assert seen["do_sample"] is True
