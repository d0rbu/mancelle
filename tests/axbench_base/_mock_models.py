"""Mock HF-style base Gemma model for axbench_base tests.

Replaces the heavy nnterp/HF stack with a deterministic stub keyed on
torch.manual_seed observations. Used by the batched-generation parity
tests in PR-2 and the integration tests in PR-4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch as t


@dataclass
class _MockModelOutput:
    logits: t.Tensor
    past_key_values: object


class _MockKVCache:
    """Minimal public-cache test double with selection-call recording."""

    def __init__(self) -> None:
        self._step = 0
        self.selection_calls: list[t.Tensor] = []

    def batch_select_indices(self, indices: t.Tensor) -> None:
        self.selection_calls.append(indices.detach().cpu().clone())

    def __repr__(self) -> str:
        return f"_MockKVCache(step={self._step})"


class MockHFGenerateModel:
    """Stub HF causal LM. The `generate()` method returns a sequence
    looked up by `t.initial_seed()` (set via the caller's preceding
    `manual_seed` call). The `__call__` method returns dummy logits and
    a KV-cache placeholder."""

    def __init__(
        self,
        seed_to_outputs: dict[int, list[int]],
        *,
        vocab_size: int = 256_000,
    ) -> None:
        self._seed_to_outputs = dict(seed_to_outputs)
        self._vocab_size = vocab_size
        self._observed_seeds: list[int] = []
        # Per-call kwargs ledger: PR-2's batched-generation parity tests
        # need to verify the kernel propagates sampling kwargs
        # (do_sample, temperature, top_p, pad_token_id, eos_token_id,
        # attention_mask, generation_config, ...) into the underlying
        # model.generate call. The mock records every kwargs dict it
        # receives so those tests can assert on propagation.
        self._observed_kwargs: list[dict[str, Any]] = []
    def generate(self, input_ids: t.Tensor, **kwargs: Any) -> t.Tensor:
        seed = int(t.initial_seed())
        self._observed_seeds.append(seed)
        self._observed_kwargs.append(dict(kwargs))
        if seed not in self._seed_to_outputs:
            raise KeyError(
                f"seed {seed} not in mock outputs map "
                f"(known: {sorted(self._seed_to_outputs)})"
            )
        continuation = self._seed_to_outputs[seed]
        # Honor max_new_tokens so PR-2's batched-generation parity tests
        # can validate that the kernel respects the kwarg. When omitted
        # (None), the full pre-registered continuation is returned.
        max_new_tokens = kwargs.get("max_new_tokens")
        if max_new_tokens is not None:
            continuation = continuation[: int(max_new_tokens)]
        out_ids = t.cat(
            [
                input_ids,
                t.tensor(
                    [continuation],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
            ],
            dim=1,
        )
        return out_ids

    def __call__(
        self,
        input_ids: t.Tensor,
        *,
        past_key_values: object = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> _MockModelOutput:
        # Return zero-logits with shape (batch, seq, vocab_size). The
        # batched-generation kernel under test should sample via
        # multinomial, but at zero-logits the multinomial draw is uniform
        # — this is fine for fixture-determinism tests because the kernel
        # respects the per-row generator.
        batch, seq = input_ids.shape
        logits_to_keep = kwargs.get("logits_to_keep", 0)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            seq = min(seq, int(logits_to_keep))
        logits = t.zeros(batch, seq, self._vocab_size, dtype=t.float32)
        return _MockModelOutput(
            logits=logits,
            past_key_values=_MockKVCache() if use_cache else None,
        )


class MockNnterpWrapper:
    """Wrapper exposing the real ``StandardizedTransformer._model`` shape."""

    def __init__(self, model: MockHFGenerateModel) -> None:
        self._model = model


class MockBaseGemma:
    """Top-level mock matching the shape of base_model returned by
    RuntimeState.base_model() — has .cfg.n_ctx, .nnterp_model, and
    .parameters()."""

    def __init__(
        self,
        seed_to_outputs: dict[int, list[int]],
        *,
        n_ctx: int = 2048,
        n_layers: int = 26,
    ) -> None:
        @dataclass
        class _Cfg:
            n_ctx: int
            n_layers: int

        self.cfg = _Cfg(n_ctx=n_ctx, n_layers=n_layers)
        hf = MockHFGenerateModel(seed_to_outputs)
        self.nnterp_model = MockNnterpWrapper(hf)

    def parameters(self):
        # Empty parameters list — sufficient for module_device best-effort.
        return iter([])


__all__ = [
    "MockBaseGemma",
    "MockHFGenerateModel",
    "MockNnterpWrapper",
    "_MockKVCache",
    "_MockModelOutput",
]
