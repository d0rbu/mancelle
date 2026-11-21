"""Smoke tests for the mock_base_gemma_for_generation fixture."""

from __future__ import annotations

import pytest
import torch as t


def test_mock_gemma_records_seeds_and_returns_specified_outputs(
    mock_base_gemma_for_generation,
):
    """Construction with a seed→outputs map. generate() returns those
    outputs in observed-seed order."""
    outputs_map = {
        7: [10, 11, 12],
        13: [20, 21, 22, 23],
    }
    mock = mock_base_gemma_for_generation(outputs_map)

    t.manual_seed(7)
    out_a = mock.nnterp_model._model.generate(
        t.tensor([[1, 2, 3]]),
        max_new_tokens=10, do_sample=True,
    )
    assert out_a.tolist() == [[1, 2, 3, 10, 11, 12]]

    t.manual_seed(13)
    out_b = mock.nnterp_model._model.generate(
        t.tensor([[5, 6]]),
        max_new_tokens=10, do_sample=True,
    )
    assert out_b.tolist() == [[5, 6, 20, 21, 22, 23]]


def test_mock_gemma_unknown_seed_raises(mock_base_gemma_for_generation):
    """Calling generate() with a seed not in the map raises clearly."""
    mock = mock_base_gemma_for_generation({7: [10, 11]})
    t.manual_seed(99)
    with pytest.raises(KeyError, match="seed 99 not in mock outputs map"):
        mock.nnterp_model._model.generate(
            t.tensor([[1, 2]]),
            max_new_tokens=10, do_sample=True,
        )


def test_mock_gemma_supports_kv_cache_call(mock_base_gemma_for_generation):
    """The mock's __call__ supports use_cache=True and returns a stub
    past_key_values that downstream code can pass back."""
    mock = mock_base_gemma_for_generation({})
    out = mock.nnterp_model._model(
        t.tensor([[1, 2, 3]]),
        use_cache=True,
    )
    assert hasattr(out, "logits")
    assert hasattr(out, "past_key_values")


def test_mock_gemma_truncates_to_max_new_tokens(
    mock_base_gemma_for_generation,
):
    """generate() with max_new_tokens=N truncates the continuation to N tokens."""
    mock = mock_base_gemma_for_generation({7: [10, 11, 12, 13, 14]})
    t.manual_seed(7)
    out = mock.nnterp_model._model.generate(
        t.tensor([[1]]),
        max_new_tokens=3, do_sample=True,
    )
    assert out.tolist() == [[1, 10, 11, 12]]


def test_mock_gemma_max_new_tokens_none_returns_full(
    mock_base_gemma_for_generation,
):
    """When max_new_tokens kwarg is missing, full continuation is returned."""
    mock = mock_base_gemma_for_generation({7: [10, 11, 12]})
    t.manual_seed(7)
    out = mock.nnterp_model._model.generate(
        t.tensor([[1]]),
        do_sample=True,
    )
    assert out.tolist() == [[1, 10, 11, 12]]
