"""Tests for device argument coercion in build_model / build_encdec_model.

Both functions accept either a string (e.g. "cpu") or a ``torch.device``
object for their ``device`` parameter. Inner helpers call ``device.type``
directly, so a plain string would raise ``AttributeError`` deep in the call
stack. W7 of axbench-eval-hardening adds a defensive coercion at the top of
each function.

These tests stub out everything after the coercion so no real checkpoint
files or GPU hardware are required.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from nanochat.checkpoint_manager import ModelPhase, build_encdec_model, build_model
from steering.conditional.eval_conditioning import EvalConditioning

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 128
_TOKENIZER_VOCAB_SIZE = _VOCAB_SIZE

# Minimal meta_data matching a pretrain checkpoint.
_PRETRAIN_META = {
    "step": 0,
    "user_config": {},
    "model_config": {
        "vocab_size": _VOCAB_SIZE,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "block_size": 64,
    },
}

# Minimal meta_data for an encdec checkpoint.
_ENCDEC_META = {
    "step": 0,
    "user_config": {
        "encoder": {"n_embd": 64, "depth": 2, "n_head": 2},
        "cross_attention": {"frequency": 1, "n_head": 2},
        "num_buckets": 8,
    },
    "model_config": {
        "vocab_size": _VOCAB_SIZE,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "block_size": 64,
    },
}


def _make_fake_state_dict():
    """Return a minimal fake state dict that satisfies key lookups."""
    # build_encdec_model inspects
    # model_data["encoder.feature_embed.weight"].shape[0] for n_sae_features.
    return {"encoder.feature_embed.weight": torch.zeros(16, 64)}


def _make_tokenizer_stub():
    tok = MagicMock()
    tok.get_vocab_size.return_value = _TOKENIZER_VOCAB_SIZE
    return tok


def _make_model_stub():
    """Return a MagicMock that satisfies the post-load model calls.

    ``unsafe=True`` is REQUIRED: since #695 ``build_encdec_model`` calls
    ``model.assert_vocab_consistent()`` after ``load_state_dict`` (``assign=True``
    replaces the persistent vocab buffer while the derived map could go stale). MagicMock
    intercepts any attribute beginning with ``assert`` and refuses it as a mistyped
    assertion helper unless ``unsafe=True`` or a spec is supplied.
    """
    mock_model = MagicMock(unsafe=True)
    mock_model.to_empty = MagicMock()
    mock_model.init_buffers = MagicMock()
    mock_model.load_state_dict = MagicMock()
    mock_model.assert_vocab_consistent = MagicMock()
    mock_model.eval = MagicMock()
    return mock_model


# Common patches that prevent real I/O / model construction in build_model.
# GPTConfig is patched here because the test meta dicts include keys
# (e.g. ``block_size``) that may not match the live dataclass signature, and
# the test is only interested in the device argument, not model construction.
_BUILD_MODEL_PATCHES = [
    "nanochat.checkpoint_manager._patch_missing_config_keys",
    "nanochat.checkpoint_manager._patch_missing_keys",
    "nanochat.checkpoint_manager._fuse_qkv_weights",
    "nanochat.checkpoint_manager.log0",
    "nanochat.checkpoint_manager.GPTConfig",
]

_BUILD_ENCDEC_PATCHES = [
    "nanochat.checkpoint_manager._patch_missing_config_keys",
    "nanochat.checkpoint_manager._patch_missing_keys",
    "nanochat.checkpoint_manager._fuse_qkv_weights",
    "nanochat.checkpoint_manager.log0",
    "nanochat.checkpoint_manager.GPTConfig",
    "nanochat.checkpoint_manager.EncoderConfig",
    "nanochat.checkpoint_manager.CrossAttentionConfig",
]

# EncoderDecoderGPT is patched SEPARATELY, with unsafe=True.
#
# Since #695, build_encdec_model calls model.assert_vocab_consistent() after
# load_state_dict — assign=True REPLACES the persistent vocab buffer while the derived
# old_to_new map keeps whatever init_buffers() produced, so they must be re-verified or
# every embedding lookup is silently wrong.
#
# MagicMock intercepts ANY attribute beginning with `assert` and refuses it as a mistyped
# assertion helper, so a vanilla patch() auto-mock cannot stand in for the real model.
_ENCDEC_MODEL_PATCH = "nanochat.checkpoint_manager.EncoderDecoderGPT"


# ---------------------------------------------------------------------------
# build_model — device coercion
# ---------------------------------------------------------------------------


class TestBuildModelDeviceCoercion:
    """build_model must coerce its ``device`` arg to torch.device before use."""

    def test_build_model_accepts_string_device(self):
        """Pass device="cpu"; captured device must be torch.device with .type == "cpu"."""
        captured = {}

        def fake_load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
            captured["device"] = device
            return {}, None, _PRETRAIN_META

        ctx_patches = [
            patch("nanochat.checkpoint_manager.load_checkpoint", side_effect=fake_load_checkpoint),
            patch("nanochat.checkpoint_manager.GPT", return_value=_make_model_stub()),
            patch("nanochat.checkpoint_manager.get_tokenizer", return_value=_make_tokenizer_stub()),
        ] + [patch(p) for p in _BUILD_MODEL_PATCHES]

        with contextlib.ExitStack() as stack:
            for cm in ctx_patches:
                stack.enter_context(cm)
            build_model("fake_dir", 0, "cpu", ModelPhase.TRAIN)

        assert isinstance(captured["device"], torch.device), (
            f"Expected torch.device, got {type(captured['device'])}"
        )
        assert captured["device"].type == "cpu"

    def test_build_model_accepts_torch_device_object(self):
        """Pass device=torch.device("cpu"); no double-wrapping should occur."""
        captured = {}
        input_device = torch.device("cpu")

        def fake_load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
            captured["device"] = device
            return {}, None, _PRETRAIN_META

        ctx_patches = [
            patch("nanochat.checkpoint_manager.load_checkpoint", side_effect=fake_load_checkpoint),
            patch("nanochat.checkpoint_manager.GPT", return_value=_make_model_stub()),
            patch("nanochat.checkpoint_manager.get_tokenizer", return_value=_make_tokenizer_stub()),
        ] + [patch(p) for p in _BUILD_MODEL_PATCHES]

        with contextlib.ExitStack() as stack:
            for cm in ctx_patches:
                stack.enter_context(cm)
            build_model("fake_dir", 0, input_device, ModelPhase.TRAIN)

        assert isinstance(captured["device"], torch.device)
        assert captured["device"] == input_device
        assert captured["device"].type == "cpu"

    def test_build_model_invalid_device_string_raises_fast(self):
        """Pass device="not-a-device"; RuntimeError must be raised at entry, NOT deep in a helper.

        After the fix, torch.device("not-a-device") raises RuntimeError before
        load_checkpoint is ever called.  Before the fix, load_checkpoint runs
        (with a plain string) and the function crashes later with AttributeError
        on ``device.type``.
        """
        load_ck_called = []

        def fake_load_checkpoint(*args, **kwargs):
            load_ck_called.append(True)
            return {}, None, _PRETRAIN_META

        with patch("nanochat.checkpoint_manager.load_checkpoint", side_effect=fake_load_checkpoint):
            with pytest.raises(RuntimeError):
                build_model("fake_dir", 0, "not-a-device", ModelPhase.TRAIN)

        assert load_ck_called == [], (
            "load_checkpoint was called, meaning the error was raised DEEP in the "
            "call stack rather than at the entry coercion point."
        )


# ---------------------------------------------------------------------------
# build_encdec_model — mirror of the above
# ---------------------------------------------------------------------------


class TestBuildEncdecModelDeviceCoercion:
    """build_encdec_model must coerce its ``device`` arg to torch.device before use."""

    def test_build_encdec_model_accepts_string_device(self):
        """Pass device="cpu"; captured device must be torch.device with .type == "cpu"."""
        captured = {}

        def fake_load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
            captured["device"] = device
            return _make_fake_state_dict(), None, _ENCDEC_META

        ctx_patches = [
            patch("nanochat.checkpoint_manager.load_checkpoint", side_effect=fake_load_checkpoint),
            patch("nanochat.checkpoint_manager.get_tokenizer", return_value=_make_tokenizer_stub()),
        ] + [patch(p) for p in _BUILD_ENCDEC_PATCHES] + [
            patch(_ENCDEC_MODEL_PATCH, return_value=_make_model_stub()),
        ]

        with contextlib.ExitStack() as stack:
            for cm in ctx_patches:
                stack.enter_context(cm)
            build_encdec_model(
                "fake_dir", 0, "cpu", ModelPhase.TRAIN,
                eval_conditioning=EvalConditioning.not_evaluating(),
            )

        assert isinstance(captured["device"], torch.device), (
            f"Expected torch.device, got {type(captured['device'])}"
        )
        assert captured["device"].type == "cpu"

    def test_build_encdec_model_accepts_torch_device_object(self):
        """Pass device=torch.device("cpu"); no double-wrapping should occur."""
        captured = {}
        input_device = torch.device("cpu")

        def fake_load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
            captured["device"] = device
            return _make_fake_state_dict(), None, _ENCDEC_META

        ctx_patches = [
            patch("nanochat.checkpoint_manager.load_checkpoint", side_effect=fake_load_checkpoint),
            patch("nanochat.checkpoint_manager.get_tokenizer", return_value=_make_tokenizer_stub()),
        ] + [patch(p) for p in _BUILD_ENCDEC_PATCHES] + [
            patch(_ENCDEC_MODEL_PATCH, return_value=_make_model_stub()),
        ]

        with contextlib.ExitStack() as stack:
            for cm in ctx_patches:
                stack.enter_context(cm)
            build_encdec_model(
                "fake_dir", 0, input_device, ModelPhase.TRAIN,
                eval_conditioning=EvalConditioning.not_evaluating(),
            )

        assert isinstance(captured["device"], torch.device)
        assert captured["device"] == input_device
        assert captured["device"].type == "cpu"

    def test_build_encdec_model_invalid_device_string_raises_fast(self):
        """Pass device="not-a-device"; RuntimeError must be raised at entry, NOT deep in a helper.

        After the fix, torch.device("not-a-device") raises RuntimeError before
        load_checkpoint is ever called.  Before the fix, load_checkpoint runs
        (with a plain string) and the function crashes later with AttributeError
        on ``device.type``.
        """
        load_ck_called = []

        def fake_load_checkpoint(*args, **kwargs):
            load_ck_called.append(True)
            return _make_fake_state_dict(), None, _ENCDEC_META

        with patch("nanochat.checkpoint_manager.load_checkpoint", side_effect=fake_load_checkpoint):
            with pytest.raises(RuntimeError):
                build_encdec_model(
                    "fake_dir", 0, "not-a-device", ModelPhase.TRAIN,
                    eval_conditioning=EvalConditioning.not_evaluating(),
                )

        assert load_ck_called == [], (
            "load_checkpoint was called, meaning the error was raised DEEP in the "
            "call stack rather than at the entry coercion point."
        )

