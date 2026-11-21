"""Shared tokenizer/model stubs for the feature-GCG BPB-Adv tests.

ONE home for the digit-word round-trip nanochat stub and the HF-style base-model
tokenizer stub, so the unit and integration tests pin the SAME production
contract (attack_rows' ``encode_prefix`` round-trip) and cannot drift apart.
"""

from __future__ import annotations

from typing import Any


class StubNanochatTokenizer:
    """Round-trip nanochat stand-in: digit words <-> int ids, method-form BOS.

    ``encode_prefix(prefix_text, tok)`` builds ``prefix_tokens`` with THIS
    tokenizer, so the derive's tokens<->text round-trip check passes exactly as
    it does on real data (where attack_rows encodes prefix_text the same way).
    ``decode`` joins ids with spaces — specials-preserving, like the real
    nanochat decode.
    """

    def get_bos_token_id(self) -> int:
        return 1

    def encode(self, text: str) -> list[int]:
        return [int(word) for word in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


class StubExtractionTokenizer:
    """HF-style tokenizer stand-in: BOS-inclusive, one id per whitespace word.

    ``tokenizer(text, padding=False)`` returns ``{"input_ids": [bos, w0, w1, ...]}``,
    mirroring the base model's own tokenizer that the DiffMean SAE forward
    consumes. ``encode(..., add_special_tokens=False)`` mirrors the no-specials
    path used to isolate DiffMean completion spans.
    """

    bos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [50 + i for i, _ in enumerate(text.split())]
        return [self.bos_token_id, *ids] if add_special_tokens else ids

    def __call__(self, text: str, padding: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [self.bos_token_id, *self.encode(text)]}


class StubBaseModel:
    """Base-Gemma stand-in for the derive.

    The DiffMean SAE forward that would consume the model is stubbed in these
    tests, but a real base model carries ``cfg.n_ctx`` (the SAE-forward
    front-clamp window) and ``.tokenizer`` (the tokenizer that forward runs
    under), so the stub mirrors that shape via :class:`StubExtractionTokenizer`.
    """

    def __init__(self, n_ctx: int = 2048) -> None:
        self.cfg: Any = type("Cfg", (), {"n_ctx": n_ctx})()
        self.tokenizer: Any = StubExtractionTokenizer()
