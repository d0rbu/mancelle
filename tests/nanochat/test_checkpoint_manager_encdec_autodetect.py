"""Tests for ``nanochat.checkpoint_manager._is_encdec_checkpoint`` and the
auto-dispatch it enables in ``load_model`` / ``load_model_from_dir``.

Why this exists
---------------
``EncoderDecoderGPT`` checkpoints usually live under
``conditional_encdec_checkpoints/<tag>/s<seg>/…`` and are loaded via the
``source="encdec/…"`` chain. The ``source.startswith("encdec")`` check
picks the right builder.

But encdec-conditioned SFT chains can save their weights under
``base_checkpoints/<tag>/…`` (e.g. ``safety_sft_conditioned_checkpoints``
in ``_SEGMENT_DIRS``). The source string for those starts with ``base``
or an SFT mixture name, so the pre-autodetect code dispatched to
``build_model`` — which strict-loaded a ``GPT``, choked on the extra
``encoder.*`` / ``cross_attn_layers.*`` state-dict keys, and crashed.

``_is_encdec_checkpoint`` peeks at the meta JSON (cheap: no tensor load)
and returns True when both ``encoder`` and ``cross_attention`` config
fields are present. The two entrypoints (``load_model`` and
``load_model_from_dir``) honor the result before picking a builder.

These tests cover:

* ``_is_encdec_checkpoint``: meta-peek behavior across all known meta
  shapes (pretrain user_config style, SFT top-level style, partial,
  missing, corrupt).
* ``load_model_from_dir``: dispatches to ``build_encdec_model`` iff the
  meta advertises encdec fields; decoder-only metas continue to go to
  ``build_model``.
* ``load_model`` (chain path): the auto-detect upgrade fires for
  decoder-rooted sources with encdec weights, and the explicit
  ``encdec/…`` path remains unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import nanochat.flash_attention as _fa_mod
from nanochat.checkpoint_manager import (
    _is_encdec_checkpoint,
    load_model,
    load_model_from_dir,
    meta_looks_encdec,
    save_checkpoint,
)
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from nanochat.gpt import GPT
from tests.conftest import (
    TINY_HEAD_DIM,
    TINY_N_EMBD,
    TINY_N_HEAD,
    TINY_N_KV_HEAD,
    TINY_N_LAYER,
    TINY_N_SAE_FEATURES,
    TINY_NUM_BUCKETS,
    TINY_SEQ_LEN,
    TINY_VOCAB,
    MockTokenizer,
    _make_encdec,
    _make_gpt,
    _make_gpt_config,
)


@pytest.fixture(autouse=True)
def _force_sdpa():
    """CPU-only tests mustn't dispatch FA3 kernels."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_tokenizer():
    """Avoid needing real tokenizer files on disk for every test."""
    mock_tok = MockTokenizer()
    with patch("nanochat.checkpoint_manager.get_tokenizer", return_value=mock_tok):
        yield


def _decoder_only_meta(step: int = 0) -> dict:
    """Build a decoder-only meta that ``build_model`` can load."""
    config = _make_gpt_config()
    return {
        "step": step,
        "model_config": {
            "sequence_len": config.sequence_len,
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_kv_head": config.n_kv_head,
            "n_embd": config.n_embd,
            "window_pattern": config.window_pattern,
        },
    }


def _encoder_field(n_embd: int = 64) -> dict:
    return {"n_embd": n_embd, "depth": 2, "n_head": 2, "n_kv_head": 2, "head_dim": 32}


def _xattn_field() -> dict:
    return {"frequency": 1}


def _pretrain_encdec_meta(step: int = 0) -> dict:
    """Pretrain encdec meta: encoder/cross_attention live under user_config."""
    m = _decoder_only_meta(step)
    m["user_config"] = {
        "encoder": _encoder_field(),
        "cross_attention": _xattn_field(),
        "num_buckets": 4,
    }
    return m


def _sft_encdec_meta(step: int = 0) -> dict:
    """SFT encdec meta: fields stored at top level (no user_config copy)."""
    m = _decoder_only_meta(step)
    m["encoder_config"] = _encoder_field()
    m["xattn_config"] = _xattn_field()
    m["num_buckets"] = 4
    return m


def _write_meta(checkpoint_dir: str, step: int, meta: dict) -> None:
    """Write just the meta file — no model_*.pt — because these tests
    only care about the JSON-peek behavior of _is_encdec_checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"), "w") as f:
        json.dump(meta, f)


# ===========================================================================
# 1. _is_encdec_checkpoint — meta peek semantics
# ===========================================================================


class TestIsEncdecCheckpoint:
    """Unit tests for the meta-peek helper. No model weights involved."""

    def test_missing_meta_returns_false(self, tmp_path: Path) -> None:
        """No meta file on disk → False (safe default; strict load will
        diagnose the actual problem)."""
        assert _is_encdec_checkpoint(str(tmp_path), step=0) is False

    def test_corrupt_meta_returns_false(self, tmp_path: Path) -> None:
        """Truncated / invalid JSON → False, not a raised JSONDecodeError."""
        bad = tmp_path / "meta_000000.json"
        bad.write_bytes(b"{not json")
        assert _is_encdec_checkpoint(str(tmp_path), step=0) is False

    def test_utf8_corrupt_meta_returns_false(self, tmp_path: Path) -> None:
        """Invalid UTF-8 bytes in the meta file → False, not a raised
        UnicodeDecodeError. The helper opens with encoding='utf-8' for
        determinism; invalid byte sequences must be swallowed the same
        way invalid JSON syntax is."""
        bad = tmp_path / "meta_000000.json"
        # 0xFF 0xFE 0xFF is not valid UTF-8 (it's a stray BOM variant).
        bad.write_bytes(b"\xff\xfe\xff")
        assert _is_encdec_checkpoint(str(tmp_path), step=0) is False

    def test_meta_path_is_directory_returns_false(self, tmp_path: Path) -> None:
        """If meta_*.json is somehow a directory (filesystem pathology),
        open() raises IsADirectoryError (OSError subclass). Swallow."""
        (tmp_path / "meta_000000.json").mkdir()
        assert _is_encdec_checkpoint(str(tmp_path), step=0) is False

    def test_explicit_utf8_encoding_used(self, tmp_path: Path) -> None:
        """Regression guard that the helper opens with encoding='utf-8'.
        Default locale encoding varies across environments; this helper
        must be deterministic. A source-string grep keeps the intent
        pinned even if the behavior test gets bypassed by a future
        refactor that inlines read() without encoding."""
        from pathlib import Path as P

        src = (P(__file__).resolve().parent.parent.parent / "nanochat" / "checkpoint_manager.py").read_text()
        # Locate the _is_encdec_checkpoint function body and confirm the
        # open() call specifies encoding.
        start = src.index("def _is_encdec_checkpoint(")
        # Next def after; grab a reasonable window.
        end = src.index("\ndef ", start + 1)
        body = src[start:end]
        assert 'encoding="utf-8"' in body or "encoding='utf-8'" in body, (
            "_is_encdec_checkpoint must open the meta file with "
            "encoding='utf-8' for cross-locale determinism"
        )

    def test_decoder_only_meta_returns_false(self, tmp_path: Path) -> None:
        """Plain model_config-only meta: no encoder/cross_attention → False."""
        _write_meta(str(tmp_path), 0, _decoder_only_meta())
        assert _is_encdec_checkpoint(str(tmp_path), step=0) is False

    def test_pretrain_style_meta_returns_true(self, tmp_path: Path) -> None:
        """Pretrain meta layout: user_config.encoder + user_config.cross_attention."""
        _write_meta(str(tmp_path), 0, _pretrain_encdec_meta())
        assert _is_encdec_checkpoint(str(tmp_path), step=0) is True

    def test_sft_style_meta_returns_true(self, tmp_path: Path) -> None:
        """SFT meta layout: top-level encoder_config + xattn_config."""
        _write_meta(str(tmp_path), 0, _sft_encdec_meta())
        assert _is_encdec_checkpoint(str(tmp_path), step=0) is True

    def test_only_encoder_no_xattn_returns_false(self, tmp_path: Path) -> None:
        """Partial meta (encoder only): we require BOTH fields to be present.
        Protects against a half-written meta that we shouldn't guess from."""
        partial = _decoder_only_meta()
        partial["user_config"] = {"encoder": _encoder_field()}
        _write_meta(str(tmp_path), 0, partial)
        assert _is_encdec_checkpoint(str(tmp_path), step=0) is False

    def test_only_xattn_no_encoder_returns_false(self, tmp_path: Path) -> None:
        """Partial meta (cross_attention only): still False."""
        partial = _decoder_only_meta()
        partial["user_config"] = {"cross_attention": _xattn_field()}
        _write_meta(str(tmp_path), 0, partial)
        assert _is_encdec_checkpoint(str(tmp_path), step=0) is False

    def test_empty_encoder_field_returns_false(self, tmp_path: Path) -> None:
        """``encoder: None`` or ``encoder: {}`` shouldn't count as encdec.
        ``bool({})`` is False, so the helper returns False by construction."""
        partial = _decoder_only_meta()
        partial["user_config"] = {"encoder": None, "cross_attention": _xattn_field()}
        _write_meta(str(tmp_path), 0, partial)
        assert _is_encdec_checkpoint(str(tmp_path), step=0) is False

        partial["user_config"] = {"encoder": {}, "cross_attention": _xattn_field()}
        _write_meta(str(tmp_path), 0, partial)
        assert _is_encdec_checkpoint(str(tmp_path), step=0) is False

    def test_step_selects_right_file(self, tmp_path: Path) -> None:
        """The helper uses meta_{step:06d}.json — not a glob. Pin this."""
        _write_meta(str(tmp_path), 100, _pretrain_encdec_meta(step=100))
        _write_meta(str(tmp_path), 200, _decoder_only_meta(step=200))
        assert _is_encdec_checkpoint(str(tmp_path), step=100) is True
        assert _is_encdec_checkpoint(str(tmp_path), step=200) is False


# ===========================================================================
# 2. load_model_from_dir — auto-dispatch to encdec builder
# ===========================================================================


def _create_decoder_checkpoint(checkpoint_dir: str, step: int = 0) -> None:
    """Full decoder-only checkpoint: model_*.pt + meta_*.json."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    config = _make_gpt_config()
    model = _make_gpt()
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None,
        _decoder_only_meta(step),
        rank=0,
    )


class TestLoadModelFromDirAutoDispatch:
    """``load_model_from_dir`` must pick the builder based on meta content,
    not on source-string shape (which this function doesn't see)."""

    def test_decoder_meta_uses_build_model(self, tmp_path: Path) -> None:
        """A plain decoder meta → build_model, not build_encdec_model."""
        ckpt_dir = tmp_path / "my_ckpts" / "d2"
        _create_decoder_checkpoint(str(ckpt_dir), step=0)

        build_model_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        build_encdec_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        with patch(
            "nanochat.checkpoint_manager.build_model", build_model_mock
        ), patch(
            "nanochat.checkpoint_manager.build_encdec_model", build_encdec_mock
        ):
            load_model_from_dir(
                str(tmp_path / "my_ckpts"),
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )
        build_model_mock.assert_called_once()
        build_encdec_mock.assert_not_called()

    def test_encdec_meta_uses_build_encdec_model(self, tmp_path: Path) -> None:
        """A meta with encoder+cross_attention → build_encdec_model."""
        ckpt_dir = tmp_path / "my_ckpts" / "d2"
        os.makedirs(ckpt_dir)
        _write_meta(str(ckpt_dir), 0, _pretrain_encdec_meta())
        # Create a placeholder model_*.pt so find_last_step succeeds.
        (ckpt_dir / "model_000000.pt").write_bytes(b"\x00")

        build_model_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        build_encdec_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        with patch(
            "nanochat.checkpoint_manager.build_model", build_model_mock
        ), patch(
            "nanochat.checkpoint_manager.build_encdec_model", build_encdec_mock
        ):
            load_model_from_dir(
                str(tmp_path / "my_ckpts"),
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )
        build_encdec_mock.assert_called_once()
        build_model_mock.assert_not_called()

    def test_sft_style_encdec_meta_uses_build_encdec_model(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: the SFT meta layout (top-level encoder_config
        + xattn_config) must also trigger encdec dispatch. This is the
        shape that nested encdec-conditioned SFT checkpoints use."""
        ckpt_dir = tmp_path / "my_ckpts" / "d2"
        os.makedirs(ckpt_dir)
        _write_meta(str(ckpt_dir), 0, _sft_encdec_meta())
        (ckpt_dir / "model_000000.pt").write_bytes(b"\x00")

        build_model_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        build_encdec_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        with patch(
            "nanochat.checkpoint_manager.build_model", build_model_mock
        ), patch(
            "nanochat.checkpoint_manager.build_encdec_model", build_encdec_mock
        ):
            load_model_from_dir(
                str(tmp_path / "my_ckpts"),
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )
        build_encdec_mock.assert_called_once()
        build_model_mock.assert_not_called()

    def test_meta_checkpoint_dir_populated_on_autodetect(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: meta_data['checkpoint_dir'] must be populated
        on the auto-detect path too, not only on the decoder path. This is
        what downstream consumers (e.g. chat_sft.py for chained saves)
        read to resolve the next-level checkpoint directory."""
        ckpt_dir = tmp_path / "my_ckpts" / "d2"
        os.makedirs(ckpt_dir)
        _write_meta(str(ckpt_dir), 0, _pretrain_encdec_meta())
        (ckpt_dir / "model_000000.pt").write_bytes(b"\x00")

        build_encdec_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        with patch(
            "nanochat.checkpoint_manager.build_encdec_model", build_encdec_mock
        ):
            _, _, meta_data = load_model_from_dir(
                str(tmp_path / "my_ckpts"),
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )
        assert meta_data["checkpoint_dir"] == str(ckpt_dir)


# ===========================================================================
# 3. load_model chain path — auto-dispatch from base/safety/etc sources
# ===========================================================================


class TestLoadModelChainAutoDispatch:
    """``load_model`` for chain sources (base, safety, sft, etc.) must
    auto-detect encdec weights stored under a base_ root. This is the
    concrete scenario the PR fixes."""

    def test_base_source_with_encdec_meta_uses_encdec_builder(
        self, tmp_path: Path
    ) -> None:
        """``source="base"`` + encdec meta at the resolved path → encdec."""
        base_dir = str(tmp_path)
        ckpt_dir = tmp_path / "base_checkpoints" / "d2"
        os.makedirs(ckpt_dir)
        _write_meta(str(ckpt_dir), 0, _pretrain_encdec_meta())
        (ckpt_dir / "model_000000.pt").write_bytes(b"\x00")

        build_model_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        build_encdec_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}), patch(
            "nanochat.checkpoint_manager.build_model", build_model_mock
        ), patch(
            "nanochat.checkpoint_manager.build_encdec_model", build_encdec_mock
        ):
            load_model(
                "base",
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )
        build_encdec_mock.assert_called_once()
        build_model_mock.assert_not_called()

    def test_safety_chain_with_encdec_meta_uses_encdec_builder(
        self, tmp_path: Path
    ) -> None:
        """``source="safety"`` + encdec meta at the nested safety SFT
        directory → encdec. Matches the
        safety_sft_conditioned_checkpoints real-world layout."""
        base_dir = str(tmp_path)
        safety_dir = (
            tmp_path / "base_checkpoints" / "d2" / "safety_sft_checkpoints"
        )
        os.makedirs(safety_dir)
        # Touch the d2 tag dir so find_largest_model is happy if called.
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        _write_meta(str(safety_dir), 0, _sft_encdec_meta())
        (safety_dir / "model_000000.pt").write_bytes(b"\x00")

        build_model_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        build_encdec_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}), patch(
            "nanochat.checkpoint_manager.build_model", build_model_mock
        ), patch(
            "nanochat.checkpoint_manager.build_encdec_model", build_encdec_mock
        ):
            load_model(
                "safety",
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )
        build_encdec_mock.assert_called_once()
        build_model_mock.assert_not_called()

    def test_base_source_with_decoder_meta_still_uses_build_model(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: pure decoder checkpoints under base_ still go
        to build_model. The auto-detect must not over-match."""
        base_dir = str(tmp_path)
        ckpt_dir = tmp_path / "base_checkpoints" / "d2"
        _create_decoder_checkpoint(str(ckpt_dir), step=0)

        build_model_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        build_encdec_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}), patch(
            "nanochat.checkpoint_manager.build_model", build_model_mock
        ), patch(
            "nanochat.checkpoint_manager.build_encdec_model", build_encdec_mock
        ):
            load_model(
                "base",
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )
        build_model_mock.assert_called_once()
        build_encdec_mock.assert_not_called()

    def test_encdec_source_remains_explicit_encdec_dispatch(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: ``source="encdec"`` still short-circuits via
        the `is_encdec = _is_encdec_root(source)` branch, never falling
        through to the chain path. The auto-detect must not double-dispatch
        or change the existing code path for explicit encdec sources."""
        base_dir = str(tmp_path)
        # Set up conditional_encdec_checkpoints/<tag>/ with an encdec meta.
        ckpt_dir = (
            tmp_path / "conditional_encdec_checkpoints" / "d2" / "s128" / "t2048.9"
        )
        os.makedirs(ckpt_dir)
        _write_meta(str(ckpt_dir), 0, _pretrain_encdec_meta())
        (ckpt_dir / "model_000000.pt").write_bytes(b"\x00")

        build_model_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        build_encdec_mock = MagicMock(return_value=(MagicMock(), MagicMock(), {}))
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}), patch(
            "nanochat.checkpoint_manager.build_model", build_model_mock
        ), patch(
            "nanochat.checkpoint_manager.build_encdec_model", build_encdec_mock
        ):
            load_model(
                "encdec",
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )
        build_encdec_mock.assert_called_once()
        build_model_mock.assert_not_called()


# ===========================================================================
# 4. Real-weights integration: auto-detect actually produces the right model
# ===========================================================================


def _real_encdec_user_config() -> dict:
    """Complete encdec user_config shaped for _make_encdec() weights."""
    return {
        "depth": TINY_N_LAYER,
        "aspect_ratio": TINY_N_EMBD // TINY_N_LAYER,
        "head_dim": TINY_HEAD_DIM,
        "max_seq_len": TINY_SEQ_LEN,
        "encoder": {
            "n_embd": TINY_N_EMBD,
            "depth": TINY_N_LAYER,
            "n_head": TINY_N_HEAD,
            "n_kv_head": TINY_N_KV_HEAD,
            "head_dim": TINY_HEAD_DIM,
        },
        "cross_attention": {"frequency": 1},
        "num_buckets": TINY_NUM_BUCKETS,
    }


def _real_encdec_model_config() -> dict:
    return {
        "sequence_len": TINY_SEQ_LEN,
        "vocab_size": TINY_VOCAB,
        "n_layer": TINY_N_LAYER,
        "n_head": TINY_N_HEAD,
        "n_kv_head": TINY_N_KV_HEAD,
        "n_embd": TINY_N_EMBD,
        "window_pattern": "L",
    }


class TestAutoDetectLoadsRealModelClass:
    """Integration tests: the auto-detect path must not just *call* the
    right builder — the loaded model must actually be the correct class
    and must accept the expected inputs. The previous unit tests mocked
    ``build_encdec_model``, so a silent regression that returned the
    wrong class (or a hand-rolled build that forgot to load weights)
    would not be caught. These tests exercise the real build path end
    to end with tiny weights."""

    def test_pretrain_style_encdec_returns_encdec_instance(
        self, tmp_path: Path
    ) -> None:
        """Save a real encdec model + pretrain-style meta, load via
        load_model_from_dir, assert it's an EncoderDecoderGPT."""
        ckpt_dir = tmp_path / "my_ckpts" / "d2"
        os.makedirs(ckpt_dir)
        model = _make_encdec()
        meta = {
            "step": 0,
            "model_config": _real_encdec_model_config(),
            "user_config": _real_encdec_user_config(),
        }
        save_checkpoint(str(ckpt_dir), 0, model.state_dict(), None, meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded, _tokenizer, loaded_meta = load_model_from_dir(
                str(tmp_path / "my_ckpts"),
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )

        assert isinstance(loaded, EncoderDecoderGPT), (
            f"auto-detect returned {type(loaded).__name__}, expected "
            "EncoderDecoderGPT"
        )
        # The checkpoint_dir field is populated on the auto-detect path.
        assert loaded_meta["checkpoint_dir"] == str(ckpt_dir)

    def test_sft_style_encdec_returns_encdec_instance(
        self, tmp_path: Path
    ) -> None:
        """SFT meta layout (top-level encoder_config / xattn_config):
        also produces a real EncoderDecoderGPT via build_encdec_model's
        ancestor walk. We set up a parent directory that carries the
        full user_config so num_buckets can be recovered."""
        parent_dir = tmp_path / "base_checkpoints" / "d2"
        child_dir = parent_dir / "safety_sft_conditioned_checkpoints"
        os.makedirs(child_dir)

        model = _make_encdec()
        parent_meta = {
            "step": 0,
            "model_config": _real_encdec_model_config(),
            "user_config": _real_encdec_user_config(),
        }
        save_checkpoint(str(parent_dir), 0, model.state_dict(), None, parent_meta)

        # Child meta: SFT-style (top-level encoder_config + xattn_config),
        # not under user_config. The child user_config is sparse.
        child_meta = {
            "step": 0,
            "model_config": _real_encdec_model_config(),
            "encoder_config": _real_encdec_user_config()["encoder"],
            "xattn_config": _real_encdec_user_config()["cross_attention"],
            "num_buckets": TINY_NUM_BUCKETS,
            "user_config": {"mixture": "safety", "source": "base"},
        }
        save_checkpoint(
            str(child_dir),
            0,
            model.state_dict(),
            None,
            child_meta,
        )

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded, _tokenizer, _loaded_meta = load_model_from_dir(
                str(parent_dir),
                t.device("cpu"),
                "eval",
                model_tag="safety_sft_conditioned_checkpoints",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )

        assert isinstance(loaded, EncoderDecoderGPT), (
            "SFT-style meta must also auto-dispatch to encdec builder"
        )

    def test_decoder_only_checkpoint_returns_gpt_instance(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: pure decoder checkpoints still produce a GPT.
        If the auto-detect ever over-matched on shape-only heuristics,
        this would catch it."""
        ckpt_dir = tmp_path / "my_ckpts" / "d2"
        os.makedirs(ckpt_dir)
        config = _make_gpt_config()
        gpt = _make_gpt()
        meta = {
            "step": 0,
            "model_config": {
                "sequence_len": config.sequence_len,
                "vocab_size": config.vocab_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_kv_head": config.n_kv_head,
                "n_embd": config.n_embd,
                "window_pattern": config.window_pattern,
            },
        }
        save_checkpoint(str(ckpt_dir), 0, gpt.state_dict(), None, meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded, _tokenizer, _loaded_meta = load_model_from_dir(
                str(tmp_path / "my_ckpts"),
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )

        assert isinstance(loaded, GPT), (
            f"decoder-only meta should produce GPT; got {type(loaded).__name__}"
        )
        assert not isinstance(loaded, EncoderDecoderGPT), (
            "Decoder-only auto-dispatch must not return EncoderDecoderGPT "
            "(EncoderDecoderGPT would be a subclass mistake — but it isn't)"
        )

    def test_loaded_encdec_accepts_encoder_inputs(
        self, tmp_path: Path
    ) -> None:
        """End-to-end: after auto-detect-load, the model accepts the
        encoder-decoder forward-pass signature without raising. This
        confirms the loaded weights match the class, not just that
        isinstance is right (a MagicMock could fake isinstance too)."""
        ckpt_dir = tmp_path / "my_ckpts" / "d2"
        os.makedirs(ckpt_dir)
        model = _make_encdec()
        meta = {
            "step": 0,
            "model_config": _real_encdec_model_config(),
            "user_config": _real_encdec_user_config(),
        }
        save_checkpoint(str(ckpt_dir), 0, model.state_dict(), None, meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded, _tokenizer, _meta = load_model_from_dir(
                str(tmp_path / "my_ckpts"),
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_manager_encdec_autodetect"),
            )

        # Forward pass with dummy encoder inputs. Shapes match the tiny
        # fixtures; this would raise on a class/weight mismatch.
        B, T = 1, 4
        idx = t.zeros(B, T, dtype=t.long)
        enc_feature_ids = t.zeros(B, 1, 1, dtype=t.long)
        enc_magnitudes = t.zeros(B, 1, 1, dtype=t.long)
        enc_mask = t.ones(B, 1, 1, dtype=t.bool)
        seg_map = t.zeros(B, T, dtype=t.long)
        loaded.eval()
        with t.no_grad():
            logits = loaded(
                idx,
                targets=None,
                enc_feature_ids=enc_feature_ids,
                enc_magnitudes=enc_magnitudes,
                enc_mask=enc_mask,
                seg_map=seg_map,
            )
        assert logits.shape[0] == B
        assert logits.shape[1] == T


# ===========================================================================
# 5. meta_looks_encdec — pure helper used by both checkpoint_manager
#    and training_progression (the latter had its own drift-prone copy)
# ===========================================================================


class TestMetaLooksEncdecPureHelper:
    """Pin the shared encoder-decoder metadata heuristic."""

    def test_pretrain_shape_returns_true(self) -> None:
        """Pretrain: user_config.encoder + user_config.cross_attention."""
        meta = {
            "user_config": {
                "encoder": {"n_embd": 64},
                "cross_attention": {"frequency": 1},
            }
        }
        assert meta_looks_encdec(meta) is True

    def test_sft_shape_returns_true(self) -> None:
        """SFT: top-level encoder_config + xattn_config."""
        meta = {
            "encoder_config": {"n_embd": 64},
            "xattn_config": {"frequency": 1},
            "num_buckets": 4,
        }
        assert meta_looks_encdec(meta) is True

    def test_plain_decoder_meta_returns_false(self) -> None:
        """Decoder-only meta: no encoder/cross_attention anywhere."""
        meta = {"model_config": {"n_layer": 12}}
        assert meta_looks_encdec(meta) is False

    def test_partial_pretrain_shape_returns_false(self) -> None:
        """Pretrain meta with only `encoder` (no `cross_attention`) → False.
        A half-written meta shouldn't be guessed from."""
        meta = {"user_config": {"encoder": {"n_embd": 64}}}
        assert meta_looks_encdec(meta) is False

    def test_partial_sft_shape_returns_false(self) -> None:
        """SFT meta with only `encoder_config` → False."""
        meta = {"encoder_config": {"n_embd": 64}, "num_buckets": 4}
        assert meta_looks_encdec(meta) is False

    def test_num_buckets_alone_returns_false(self) -> None:
        """Regression guard against the old permissive heuristic: a meta
        with only top-level ``num_buckets`` (no encoder/cross_attention
        anywhere) used to return True under
        ``_metadata_looks_encoder_decoder`` and risk mis-dispatching a
        decoder meta with a stray num_buckets field to
        build_encdec_model. The consolidated heuristic returns False."""
        meta = {"num_buckets": 4, "model_config": {"n_layer": 12}}
        assert meta_looks_encdec(meta) is False

    def test_n_sae_features_alone_returns_false(self) -> None:
        """The old ``_metadata_looks_encoder_decoder`` checked for
        ``n_sae_features`` at top level, but no save path actually
        writes it there. Verify that a meta with only this (hypothetical)
        key is NOT treated as encdec."""
        meta = {"n_sae_features": 16384, "model_config": {"n_layer": 12}}
        assert meta_looks_encdec(meta) is False

    def test_empty_meta_returns_false(self) -> None:
        assert meta_looks_encdec({}) is False


class TestNoDuplicateMetaPeek:
    """Prevent regression: training_progression must not reintroduce a private
    metadata classifier."""

    def test_training_progression_source_imports_meta_looks_encdec(
        self,
    ) -> None:
        """Source-string guard against another drift-prone helper."""
        src = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "train"
            / "training_progression.py"
        ).read_text()
        assert "meta_looks_encdec" in src, (
            "scripts/train/training_progression.py must import meta_looks_encdec "
            "from nanochat.checkpoint_manager — a duplicate meta-peek "
            "function would drift from the canonical heuristic"
        )
        assert "_metadata_looks_encoder_decoder" not in src
        # Also guard that the old permissive heuristic isn't re-introduced.
        assert '"n_sae_features" in meta' not in src, (
            "The old _metadata_looks_encoder_decoder heuristic including "
            "`n_sae_features in meta` has been removed — don't restore it."
        )
