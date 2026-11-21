"""Tests for ``scripts/train/base_train.py`` resume init-path.

Slice 11 audit P1 #8: the original sequence ``init_weights() ->
load_state_dict(assign=True)`` ran a full normal_/uniform_-init pass
on every parameter only to immediately overwrite them via
``load_state_dict(assign=True)``. The fix calls ``init_buffers()``
post-load instead (only the non-persistent rotary cos/sin need to be
computed; params come from the checkpoint).

Two complementary tests:

1. **Source-inspection guard** — pins the structure of the resume
   branch so a future "clean up the conditional" refactor doesn't
   silently restore ``init_weights()``-then-overwrite.

2. **Roundtrip equivalence test** — builds a tiny GPT, applies the
   refactored sequence (``to_empty -> load_state_dict -> init_buffers``)
   and asserts that the resulting tensors are bit-identical to those
   produced by the pre-fix sequence (``to_empty -> init_weights ->
   load_state_dict``). This is the actual correctness contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from nanochat.gpt import GPT, GPTConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_TRAIN_PATH = REPO_ROOT / "scripts" / "train" / "base_train.py"


# ---------------------------------------------------------------------------
# Source guard
# ---------------------------------------------------------------------------


def test_resume_branch_uses_init_buffers_not_init_weights() -> None:
    """The resume branch must call ``init_buffers`` after
    ``load_state_dict``, NOT ``init_weights`` before."""
    src = BASE_TRAIN_PATH.read_text()
    # The resume branch is identified by the ``if resuming:`` line.
    resume_match = re.search(r"^if resuming:\s*$", src, re.MULTILINE)
    assert resume_match is not None, "resume branch not found"
    # Capture ~30 lines following the if-resuming line.
    snippet = src[resume_match.start() : resume_match.start() + 1200]
    assert "init_buffers" in snippet, (
        "scripts/train/base_train.py resume branch must call "
        "``model.init_buffers()`` after load_state_dict (slice 11 audit P1 #8)."
    )
    # Confirm the pre-resume init_weights is gated on NOT resuming.
    assert re.search(r"else:\s*\n\s*model\.init_weights\(\)", src), (
        "scripts/train/base_train.py must gate ``init_weights()`` on "
        "the not-resuming branch (slice 11 audit P1 #8): otherwise we "
        "waste compute initializing weights only to overwrite them."
    )


# ---------------------------------------------------------------------------
# Roundtrip equivalence
# ---------------------------------------------------------------------------


def _build_tiny_meta() -> GPT:
    """Build a tiny GPT on the meta device for shape-only construction."""
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    return model


def _build_initialized_state_dict() -> dict[str, torch.Tensor]:
    """Build a fresh GPT and return its parameter state_dict + buffer
    snapshots, materialized on CPU."""
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_refactored_resume_path_matches_prefix_path_post_load() -> None:
    """Both sequences must produce identical post-load model tensors.

    Pre-fix sequence (the wasteful one):
        to_empty(device='cpu')
        init_weights()                  <- normal_/uniform_-inits everything
        load_state_dict(state_dict, assign=True)   <- overwrites params

    Post-fix sequence (the lean one):
        to_empty(device='cpu')
        load_state_dict(state_dict, assign=True)
        init_buffers()                  <- only rotary cos/sin

    Contract: every tensor in ``model.state_dict()`` AND every named
    buffer (including non-persistent ones) must match between the two.
    """
    state_dict = _build_initialized_state_dict()

    # Pre-fix model
    pre_fix = _build_tiny_meta()
    pre_fix.to_empty(device="cpu")
    pre_fix.init_weights()
    pre_fix.load_state_dict(state_dict, strict=True, assign=True)

    # Post-fix model
    post_fix = _build_tiny_meta()
    post_fix.to_empty(device="cpu")
    post_fix.load_state_dict(state_dict, strict=True, assign=True)
    post_fix.init_buffers()

    # Compare all named parameters
    pre_params = dict(pre_fix.named_parameters())
    post_params = dict(post_fix.named_parameters())
    assert set(pre_params.keys()) == set(post_params.keys())
    for name, pre_param in pre_params.items():
        torch.testing.assert_close(
            pre_param,
            post_params[name],
            msg=f"param {name!r} diverged between pre-fix and post-fix init paths",
        )

    # Compare all named buffers, including non-persistent ones (cos/sin)
    pre_buffers = dict(pre_fix.named_buffers())
    post_buffers = dict(post_fix.named_buffers())
    assert set(pre_buffers.keys()) == set(post_buffers.keys())
    for name, pre_buffer in pre_buffers.items():
        torch.testing.assert_close(
            pre_buffer,
            post_buffers[name],
            msg=f"buffer {name!r} diverged between pre-fix and post-fix init paths",
        )
