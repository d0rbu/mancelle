"""BOS-safe prompt-token editing helpers for AxBench-BPB-Adv.

Generic edit primitives live in
``steering.conditional.feature_gcg.edit_primitives``.
"""

from __future__ import annotations

from collections.abc import Sequence

from steering.conditional.feature_gcg.edit_primitives import EditableSpan as _EditableSpan
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError

__all__ = [
    "replace_editable_tokens",
    "select_final_k_editable_span",
]


def select_final_k_editable_span(
    prompt_tokens: Sequence[int],
    *,
    k: int,
) -> _EditableSpan:
    """Select the final ``k`` non-BOS prompt tokens.

    ``k=0`` is a valid no-adversary control and returns an empty span at the
    end of the prompt. BOS is position 0 and is never editable.
    """

    if k < 0:
        raise AxBenchBpbAdvError(f"k must be >= 0, got {k}")
    if len(prompt_tokens) < 1:
        raise AxBenchBpbAdvError("prompt_tokens must contain at least a BOS token")
    if k == 0:
        return _EditableSpan(start=len(prompt_tokens), count=0)

    non_bos_count = len(prompt_tokens) - 1
    if non_bos_count < k:
        raise AxBenchBpbAdvError(
            f"prompt has only {non_bos_count} non-BOS tokens, fewer than k={k}"
        )
    start = len(prompt_tokens) - k
    if start <= 0:
        raise AxBenchBpbAdvError(
            f"editable span would include BOS: start={start}, count={k}"
        )
    return _EditableSpan(start=start, count=k)


def replace_editable_tokens(
    prompt_tokens: Sequence[int],
    span: _EditableSpan,
    replacement_tokens: Sequence[int],
) -> list[int]:
    if span.start <= 0 and span.count > 0:
        raise AxBenchBpbAdvError(
            f"editable span must start after BOS, got start={span.start}"
        )
    if span.count < 0:
        raise AxBenchBpbAdvError(f"editable span count must be >= 0, got {span.count}")
    if span.stop > len(prompt_tokens):
        raise AxBenchBpbAdvError(
            f"editable span [{span.start}, {span.stop}) exceeds prompt length "
            f"{len(prompt_tokens)}"
        )
    if len(replacement_tokens) != span.count:
        raise AxBenchBpbAdvError(
            f"replacement length {len(replacement_tokens)} must equal editable "
            f"span count {span.count}"
        )
    out = [int(token_id) for token_id in prompt_tokens]
    out[span.start:span.stop] = [int(token_id) for token_id in replacement_tokens]
    if len(out) != len(prompt_tokens):
        raise AxBenchBpbAdvError("replacement changed prompt token length")
    return out
