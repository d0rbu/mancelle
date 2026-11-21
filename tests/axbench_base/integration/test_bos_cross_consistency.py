"""Guard the continuation protocol's text-only BOS ownership boundary."""

from __future__ import annotations

import inspect

from steering.eval.axbench_base.synth import continuation_protocol


def test_stage1_continuation_parser_is_text_only_no_bos_injection():
    """Static guard: the authenticated continuation parser truncates
    an LLM response and decodes it back to a STRING -- it is never forwarded
    through the base model + SAE, so (unlike the 5 sites fixed by Bug 1) it
    must NOT inject a BOS. A BOS token decodes to a visible string fragment,
    so "fixing" this site by copy-pasting the BOS-injection pattern would
    corrupt every synth continuation with a stray leading token.

    Mutation-proof by construction: asserts on the FUNCTION'S OWN source (via
    ``inspect.getsource``, not a line number or the whole file). The function
    MAY reference ``resolve_bos_token_id`` / ``special_token_ids`` to DETECT an
    LLM-emitted special token and RAISE (the Q2 refusal) — that is not BOS
    injection — so the guard checks for the codebase's actual BOS-prepend
    list-literal patterns instead, and pins that the function returns a decoded
    STRING (never a BOS-prepended ``model_input`` forwarded through the model).
    """
    source = inspect.getsource(
        continuation_protocol.parse_stage1_continuation_response
    )
    assert "add_special_tokens=False" in source
    assert "tokenizer.decode(" in source  # returns TEXT, not a forwarded input
    # None of the BOS-injection list-prepend patterns used by the 5 Bug-1 sites
    # (extraction._bos_prepended_model_input, qc.get_many, qc._tokenize_prefix_and_full,
    # feature_diffmean) may appear here.
    for inject_pattern in ("[bos_id] +", "[int(bos_id)] +", "[int(bos_id),", "[bos_id,"):
        assert inject_pattern not in source, (
            "parse_stage1_continuation_response must not inject a BOS; found the "
            f"prepend pattern {inject_pattern!r} in its source."
        )
