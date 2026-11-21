"""Shared fixtures for tests/profiling — toy CPU-only stubs of a progression run."""

from __future__ import annotations

import pytest
import torch


class _ToyTokenizer:
    """Minimal tokenizer satisfying nanochat's progression-eval call sites."""

    def __init__(self) -> None:
        self._bos = 0

    def get_bos_token_id(self) -> int:
        return self._bos

    def encode(self, text: str, prepend: int | str | None = None) -> list[int]:
        toks = [ord(ch) % 128 for ch in text[:16]] or [1]
        if isinstance(prepend, int):
            toks = [prepend, *toks]
        elif prepend == "<bos>":
            toks = [self._bos, *toks]
        return toks

    def __call__(
        self,
        texts: str | list[str],
        prepend: int | str | None = None,
    ) -> list[int] | list[list[int]]:
        if isinstance(texts, str):
            return self.encode(texts, prepend=prepend)
        return [self.encode(t, prepend=prepend) for t in texts]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(max(1, int(t) % 128)) for t in tokens)

    def render_for_completion(self, conversation: dict) -> list[int]:
        return [self._bos, 1, 2, 3]


@pytest.fixture
def toy_tokenizer() -> _ToyTokenizer:
    return _ToyTokenizer()


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")
