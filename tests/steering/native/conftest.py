"""Tiny real HF Gemma fixtures for native-collector tests (CPU, no download).

Two fixture flavours:

* ``tiny_gemma2`` / ``tiny_gemma3`` — session-scoped SINGLE instances. Cheap
  for read-only tests, but MUST NOT be passed to ``build_native_model`` (the
  truncation build mutates ``.layers``/``.norm``/``config`` IN PLACE).
* ``tiny_gemma2_factory`` / ``tiny_gemma3_factory`` — callables returning a
  FRESH seeded twin per call (same seed + ``fork_rng`` ⇒ byte-identical
  weights to the session fixture and to every other twin). Use these for any
  test that builds a truncated collector or needs an independent mutable copy
  (truncation MUTATES in place). This is the shared twin/factory pattern the
  truncated-collector tests use — re-pointed hook tests reuse these instead of
  copy-pasting the construction.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch as t
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
)

TINY_VOCAB = 64
TINY_D_MODEL = 32
TINY_LAYERS = 4
TINY_HEADS = 4
TINY_KV_HEADS = 2
TINY_HEAD_DIM = 8
TINY_FFN = 64
TINY_CTX = 64

# --- Mixed-layer_types tiny gemma-3 (production-realism fixture) ----------- #
# The uniform `_build_tiny_gemma3` above derives `layer_types =
# ['sliding_attention'] * 4` (verified: with `sliding_window_pattern=6` and
# 4 layers, `(i+1) % 6` is truthy for every i, so NO `full_attention` layer
# exists). That never exercises gemma-3's PRODUCTION reality — an INTERLEAVED
# full/sliding stack with DISTINCT per-type rope and a real (small)
# `sliding_window`. This second flavour restores that:
#
#   num_hidden_layers=8 + sliding_window_pattern=6  (the installed
#   `Gemma3TextConfig.__post_init__` derivation: "sliding_attention" if
#   `(i+1) % sliding_window_pattern` else "full_attention")
#   ⇒ layer_types = [sliding ×5, full @5, sliding ×2]   (VERIFIED below).
#
# So index 5 is the ONLY `full_attention` layer; every other index is
# `sliding_attention`. This lets the truncation==reference test parametrize
# over a kept slice that (a) STILL contains the full layer (layer >= 5 →
# `set(layer_types[:kept])` = {sliding, full}) AND (b) drops ALL full layers
# (layer < 5 → `set(...)` collapses to {sliding} only — the installed
# `Gemma3TextModel.forward` builds `position_embeddings` per
# `set(self.config.layer_types)` and indexes both the per-type rotary and the
# per-type causal mask by `config.layer_types[i]`, so a `layer_types`
# slice-misalignment / full-vs-sliding bug genuinely changes the output —
# proven non-vacuous: full@5 vs sliding@5 differ by max_abs≈0.34 at seq>sw,
# and an off-by-one `layer_types` slice raises IndexError).
#
# DISTINCT per-type rope: `rope_theta` sets the `full_attention` rope base and
# `rope_local_base_freq` sets the `sliding_attention` rope base (installed
# `Gemma3TextConfig.convert_rope_params_to_dict`). They default to 1e6 vs 1e4
# (already distinct); we pin them explicitly and FAR apart so any
# full-vs-sliding rope mix-up changes the output. `sliding_window` is set
# small (TINY_MIXED_SLIDING_WINDOW=4) so a seq longer than it makes the
# sliding-window mask actually differ from the full causal mask.
TINY_MIXED_LAYERS = 8
TINY_MIXED_SLIDING_WINDOW = 4
TINY_MIXED_SWP = 6  # sliding_window_pattern ⇒ full_attention at index 5 only
TINY_MIXED_FIRST_FULL_IDX = 5  # the single full_attention layer's index
TINY_MIXED_ROPE_THETA_FULL = 1_000_000.0  # full_attention rope base
TINY_MIXED_ROPE_THETA_SLIDING = 10_000.0  # sliding_attention rope base (≠ full)

# Single seed shared by the session fixtures AND the factories so a fresh twin
# == the session fixture's weights (and any two twins are identical).
TINY_SEED = 20260516


def _build_tiny_gemma2() -> Gemma2ForCausalLM:
    """A fresh seeded tiny Gemma2ForCausalLM (the canonical construction)."""
    with t.random.fork_rng():
        t.manual_seed(TINY_SEED)
        cfg = Gemma2Config(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_D_MODEL,
            intermediate_size=TINY_FFN,
            num_hidden_layers=TINY_LAYERS,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            head_dim=TINY_HEAD_DIM,
            max_position_embeddings=TINY_CTX,
            attn_implementation="eager",  # ty: ignore[unknown-argument]
        )
        return Gemma2ForCausalLM(cfg).to(t.float32).eval()  # ty: ignore[invalid-argument-type]


def _build_tiny_gemma3() -> Gemma3ForCausalLM:
    """A fresh seeded tiny Gemma3ForCausalLM (the canonical construction).

    Gemma3ForCausalLM takes Gemma3TextConfig directly (not the outer
    Gemma3Config which is multimodal-only). Gemma3TextConfig accepts the same
    flat dimension kwargs as Gemma2Config; it auto-derives layer_types for the
    4-layer tiny model (all sliding_attention, correct for eager mode).
    """
    with t.random.fork_rng():
        t.manual_seed(TINY_SEED)
        cfg = Gemma3TextConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_D_MODEL,
            intermediate_size=TINY_FFN,
            num_hidden_layers=TINY_LAYERS,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            head_dim=TINY_HEAD_DIM,
            max_position_embeddings=TINY_CTX,
            attn_implementation="eager",  # ty: ignore[unknown-argument]
        )
        return Gemma3ForCausalLM(cfg).to(t.float32).eval()  # ty: ignore[invalid-argument-type]


def _build_tiny_gemma3_mixed() -> Gemma3ForCausalLM:
    """A fresh seeded tiny Gemma3ForCausalLM with INTERLEAVED layer_types.

    Mirrors `_build_tiny_gemma3`'s seeding/size/dtype verbatim (same
    TINY_SEED, fork_rng, eager, fp32) but with `num_hidden_layers=8` and an
    explicit `sliding_window_pattern=6`, so the installed
    `Gemma3TextConfig.__post_init__` derives

        layer_types = ['sliding_attention'] * 5
                     + ['full_attention']        # index 5 (the ONLY full)
                     + ['sliding_attention'] * 2

    (asserted below), plus a small `sliding_window=4` and DISTINCT per-type
    rope bases (full=1e6 via `rope_theta`, sliding=1e4 via
    `rope_local_base_freq`). This is the production reality the uniform
    all-`sliding_attention` tiny gemma-3 never exercises. Tiny / CPU / seeded
    / no network, exactly like the other builders.
    """
    with t.random.fork_rng():
        t.manual_seed(TINY_SEED)
        cfg = Gemma3TextConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_D_MODEL,
            intermediate_size=TINY_FFN,
            num_hidden_layers=TINY_MIXED_LAYERS,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            head_dim=TINY_HEAD_DIM,
            max_position_embeddings=TINY_CTX,
            sliding_window=TINY_MIXED_SLIDING_WINDOW,
            sliding_window_pattern=TINY_MIXED_SWP,  # ty: ignore[unknown-argument]
            rope_theta=TINY_MIXED_ROPE_THETA_FULL,  # ty: ignore[unknown-argument]
            rope_local_base_freq=TINY_MIXED_ROPE_THETA_SLIDING,  # ty: ignore[unknown-argument]
            attn_implementation="eager",  # ty: ignore[unknown-argument]
        )
        # Fail LOUDLY here if a transformers upgrade changes the derivation —
        # the whole point of this fixture is the interleaved full/sliding
        # reality; a silent regression to uniform would make the realism gate
        # vacuous. (Belt-and-braces: the dedicated fixture-shape test also
        # asserts this.)
        expected = (
            ["sliding_attention"] * TINY_MIXED_FIRST_FULL_IDX
            + ["full_attention"]
            + ["sliding_attention"]
            * (TINY_MIXED_LAYERS - TINY_MIXED_FIRST_FULL_IDX - 1)
        )
        assert cfg.layer_types == expected, (
            "tiny mixed gemma-3 layer_types regressed to "
            f"{cfg.layer_types!r}; expected interleaved {expected!r} "
            "(transformers Gemma3TextConfig derivation changed)"
        )
        return Gemma3ForCausalLM(cfg).to(t.float32).eval()  # ty: ignore[invalid-argument-type]


@pytest.fixture(scope="session")
def tiny_gemma2():
    return _build_tiny_gemma2()


@pytest.fixture(scope="session")
def tiny_gemma3():
    return _build_tiny_gemma3()


@pytest.fixture(scope="session")
def tiny_gemma3_mixed():
    """Session-scoped SINGLE interleaved-layer_types tiny gemma-3.

    Read-only use only — like the other session fixtures it MUST NOT be passed
    to a truncating build (the transform mutates `.layers`/`.norm`/`config` in
    place); use `tiny_gemma3_mixed_factory` for that.
    """
    return _build_tiny_gemma3_mixed()


@pytest.fixture(scope="session")
def tiny_gemma2_factory() -> Callable[[], Gemma2ForCausalLM]:
    """Returns a callable building a fresh seeded tiny Gemma2 twin per call.

    Session-scoped (the *callable* is reusable); each invocation builds an
    independent mutable model with byte-identical weights — required because
    the truncation build mutates the model in place.
    """
    return _build_tiny_gemma2


@pytest.fixture(scope="session")
def tiny_gemma3_factory() -> Callable[[], Gemma3ForCausalLM]:
    """Returns a callable building a fresh seeded tiny Gemma3 twin per call."""
    return _build_tiny_gemma3


@pytest.fixture(scope="session")
def tiny_gemma3_mixed_factory() -> Callable[[], Gemma3ForCausalLM]:
    """Returns a callable building a fresh seeded INTERLEAVED tiny gemma-3 twin.

    Same shared twin/factory pattern as the other factories: every invocation
    builds an independent mutable model with byte-identical weights (same
    TINY_SEED + fork_rng), required because the truncation build mutates the
    model in place. Used by the mixed-layer_types truncation==reference gate.
    """
    return _build_tiny_gemma3_mixed
