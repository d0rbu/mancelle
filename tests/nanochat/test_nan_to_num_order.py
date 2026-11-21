"""Slice-09 audit: ``nan_to_num`` order documentation (manifest §7 slice-9 row).

The audit added an in-source rationale block immediately above the pre-SDPA
``q.nan_to_num()`` / ``k.nan_to_num()`` calls explaining:
1. Why padding zeros become NaN (RMSNorm of all-zero vector).
2. Why pre-SDPA cleanup is required (softmax NaN propagation).
3. Why post-SDPA cleanup is required (all-masked rows).
4. Why post-c_proj cleanup is unnecessary (linear map preserves finiteness).

A future cleanup that strips the rationale would silently invite the same
debugging effort that motivated the original fix. This regression guard
re-asserts the comments are still present.
"""

import inspect

from steering.conditional.encoder_decoder.cross_attention import CrossAttentionLayer


def _forward_source() -> str:
    return inspect.getsource(CrossAttentionLayer.forward)


def test_pre_sdpa_nan_rationale_present() -> None:
    src = _forward_source()
    # Headline marker so future maintainers can grep for it.
    assert "NaN handling order" in src
    # Key talking points: pre-SDPA cleanup AND post-SDPA cleanup AND
    # the absence of a post-c_proj cleanup.
    assert "MUST clean q and k BEFORE SDPA" in src
    assert "MUST clean y AFTER SDPA" in src
    assert "do NOT clean after c_proj" in src


def test_pre_and_post_nan_to_num_calls_still_present() -> None:
    """The semantic guarantee — q/k cleaned pre-SDPA, y cleaned post-SDPA — is
    enforced by `tests/nanochat/test_cross_attention.py::TestNaNHandlingOrder`
    via source inspection. We replicate the explicit-call check here so a
    rewrite that loses *both* the calls and the comment block is caught."""
    src = _forward_source()
    assert "q = q.nan_to_num()" in src
    assert "k = k.nan_to_num()" in src
    assert "y = y.nan_to_num()" in src


def test_rationale_references_rmsnorm_origin() -> None:
    """The comment must mention RMSNorm as the source of NaN so a reader who
    deletes 'nan_to_num because we filter scatter zeros' understands the
    upstream cause."""
    src = _forward_source()
    assert "RMSNorm" in src or "RMS norm" in src
