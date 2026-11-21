"""Frozen scalar GCG candidate validation from 4e3fb993e50d5f0685fb4380aa619c2e723a503c.

Only the function name/type annotation and module-helper bindings differ from
that reference. Keep the validation body independent of production changes.
"""

from __future__ import annotations

import sys
import unicodedata
from typing import Any


def scalar_candidate(self: Any, token_ids: tuple[int, ...]) -> tuple[str, str | None]:
    module = sys.modules[type(self).__module__]
    _encode = module._encode
    _decode = module._decode
    if len(token_ids) != len(self.seed_token_ids):
        return "", f"candidate must contain exactly {len(self.seed_token_ids)} token ids"
    if any(token_id not in self._allowed_token_id_set for token_id in token_ids):
        return "", "candidate contains a token outside the allowed pool"
    try:
        text = _decode(self.runtime.tokenizer, token_ids)
        if (
            not text
            or text.strip() != text
            or "\x00" in text
            or unicodedata.normalize("NFC", text) != text
        ):
            return (
                text,
                "candidate text is blank, outer-spaced, contains NUL, or changes under NFC",
            )
        if _encode(self.runtime.tokenizer, text, add_special_tokens=False) != token_ids:
            return text, "candidate decode/re-encode changes optimized token ids"
        for pair in self.layouts:
            for side in (pair.positive, pair.negative):
                rendered = f"{side.content}{self.separator}{text}"
                wrapper = self._wrapper
                if wrapper is None:
                    return "", "candidate wrapper is unavailable"
                observed = _encode(
                    self.runtime.tokenizer,
                    f"{wrapper.prefix}{rendered}{wrapper.suffix}",
                    add_special_tokens=True,
                )
                if observed != side.fixed_before + token_ids + side.fixed_after:
                    return text, "candidate changes a fixed rendered wrapper or tail"
        return text, None
    except (UnicodeError, ValueError, TypeError, RuntimeError) as exc:
        return "", f"candidate canonicalization failed: {exc}"
