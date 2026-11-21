"""Test-only reference for the tri-state QC equivalence property (issue #518).

``fc_active_on_continuation`` was negative_mode's original strict-zero QC predicate. After
issue #518, negative_mode uses ``classify_avoid_continuation``; this reference is retained ONLY
as the independent boolean implementation that
``test_classify_equivalent_to_old_two_function_combo`` differentially pins
``classify_avoid_continuation`` against. It is NOT production code. It deliberately mirrors the
removed ``qc.fc_active_on_continuation`` (minus debug logging and the profiling counter), calling
the same private qc helpers so the equivalence property's tokenizer/extractor monkeypatches (on
the ``qc`` module) still apply.
"""

from __future__ import annotations

from typing import Any

from steering.eval.axbench_base.synth import qc
from steering.eval.axbench_base.synth.response_parser import (
    DEFAULT_EXTRACTION_TOKENIZER_ID,
)


def fc_active_on_continuation(
    prefix_text: str,
    continuation_text: str,
    f_c: int,
    base_model: Any,
    base_sae: Any,
    *,
    sae_layer: int,
    extraction_threshold: float = 0.0,
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
) -> bool:
    """Reference: True iff ``f_c`` fires above ``extraction_threshold`` in the continuation slice.

    Returns ``False`` for the NO_TOKENS case (``1 + n_prefix >= len(full_tokens)``) — the
    original behavior whose conflation of NO_TOKENS with INACTIVE was the issue-#518 bug.

    ``qc._tokenize_prefix_and_full`` prepends exactly one leading BOS to ``full_tokens``
    (position 0), so both the NO_TOKENS boundary and the continuation slice below carry a
    ``+1`` offset to exclude it -- mirroring ``qc.classify_avoid_continuation``.
    """
    prefix_tokens, full_tokens = qc._tokenize_prefix_and_full(
        prefix_text, continuation_text, tokenizer_id=tokenizer_id,
    )
    n_prefix = len(prefix_tokens)
    if 1 + n_prefix >= len(full_tokens):
        return False
    acts = qc._run_sae_on_full(full_tokens, base_model, base_sae, sae_layer=sae_layer)
    continuation_fc = acts[1 + n_prefix:, f_c]
    return bool((continuation_fc > extraction_threshold).any().item())
