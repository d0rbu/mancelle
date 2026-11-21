"""Tests for steering.eval.eval_cache_paths.default_target_cache_dir.

The path layout mirrors steering.feature_paths so operators recognize the
component nesting. Caches live under nanochat_cache/eval_caches/ by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

import steering.eval.eval_cache_paths as eval_cache_paths
from steering.eval.context_cache_io import HASH_TRUNCATION_LEN, CacheIdentity
from steering.eval.eval_cache_paths import (
    _parse_sae_id,
    default_target_cache_dir,
)
from steering.refined_types import PositiveInt

_PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
_CONTEXT_LEN = PositiveInt.parse(1_024)
_LONG_CONTEXT_LEN = PositiveInt.parse(2_048)

# ---------------------------------------------------------------------------
# _parse_sae_id
# ---------------------------------------------------------------------------


def test_parse_sae_id_residual_16k():
    act, width = _parse_sae_id("gemmascope-residual-l8-16k")
    assert act == "res"
    assert width == 16384


def test_parse_sae_id_mlp_65k():
    act, width = _parse_sae_id("gemmascope-mlp-l8-65k")
    assert act == "mlp"
    assert width == 65536


def test_parse_sae_id_attention_16k():
    act, width = _parse_sae_id("gemmascope-attention-l12-16k")
    assert act == "att"
    assert width == 16384


def test_parse_sae_id_attention_short_alias():
    """The short 'att' alias should also parse so operators can use either."""
    act, width = _parse_sae_id("gemmascope-att-l12-16k")
    assert act == "att"
    assert width == 16384


def test_parse_sae_id_residual_short_alias():
    """The 'res' short alias also parses."""
    act, width = _parse_sae_id("gemmascope-res-l8-16k")
    assert act == "res"
    assert width == 16384


def test_parse_sae_id_residual_131k():
    act, width = _parse_sae_id("gemmascope-residual-l12-131k")
    assert act == "res"
    assert width == 131072


def test_parse_sae_id_residual_524k():
    act, width = _parse_sae_id("gemmascope-residual-l12-524k")
    assert act == "res"
    assert width == 524288


def test_parse_sae_id_residual_1m():
    act, width = _parse_sae_id("gemmascope-residual-l12-1m")
    assert act == "res"
    assert width == 1048576


def test_parse_sae_id_unknown_format_raises():
    with pytest.raises(ValueError, match="cannot parse"):
        _parse_sae_id("not-a-known-format")


def test_parse_sae_id_unknown_activation_raises():
    with pytest.raises(ValueError, match="activation"):
        _parse_sae_id("gemmascope-bogus-l8-16k")


def test_parse_sae_id_unknown_width_raises():
    with pytest.raises(ValueError, match="width"):
        _parse_sae_id("gemmascope-residual-l8-99x")


def test_parse_sae_id_unknown_layer_raises():
    with pytest.raises(ValueError, match="layer"):
        _parse_sae_id("gemmascope-residual-lXX-16k")


# ---------------------------------------------------------------------------
# default_target_cache_dir — happy paths
# ---------------------------------------------------------------------------


def _make_identity(
    sae_id="gemmascope-residual-l8-16k",
    feature_selection_hash="a" * 64,
):
    return CacheIdentity(
        model_id="gemma-2-2b",
        sae_id=sae_id,
        layer=8,
        pooling="max_pool",
        feature_selection_hash=feature_selection_hash,
    )


def test_default_target_cache_dir_chat_target_layout(tmp_path):
    identity = _make_identity()
    p = default_target_cache_dir(
        suite="chat_target",
        sae_identity=identity,
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=tmp_path,
    )
    parts = p.parts
    # Expected: <tmp>/eval_caches/chat_target/m_gemma-2-2b/l8/a_res/p_max_pool/f16384/c1024/<hash[:16]>
    assert "eval_caches" in parts
    assert "chat_target" in parts
    assert "m_gemma-2-2b" in parts
    assert "l8" in parts
    assert "a_res" in parts
    assert "p_max_pool" in parts
    assert "f16384" in parts
    assert "c1024" in parts
    # Hash truncated to first 16 chars.
    assert parts[-1] == "a" * HASH_TRUNCATION_LEN


def test_default_target_cache_dir_core_target_layout(tmp_path):
    p = default_target_cache_dir(
        suite="core_target",
        sae_identity=_make_identity(),
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=tmp_path,
    )
    assert "core_target" in p.parts


def test_default_target_cache_dir_samples_layout(tmp_path):
    p = default_target_cache_dir(
        suite="samples",
        sae_identity=_make_identity(),
        model_max_seq_len=_LONG_CONTEXT_LEN,
        base_dir=tmp_path,
    )
    assert "samples" in p.parts
    assert "c2048" in p.parts


def test_default_target_cache_dir_harmfulness_layout(tmp_path):
    p = default_target_cache_dir(
        suite="harmfulness",
        sae_identity=_make_identity(),
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=tmp_path,
    )
    assert "harmfulness" in p.parts


def test_default_target_cache_dir_uses_get_base_dir_when_base_dir_none(monkeypatch):
    """When base_dir is None, the helper must read nanochat.common.get_base_dir."""
    import steering.eval.eval_cache_paths as mod

    fake_root = "/tmp/test_get_base_dir_xyz"
    monkeypatch.setattr(mod, "get_base_dir", lambda: fake_root)
    p = default_target_cache_dir(
        suite="chat_target",
        sae_identity=_make_identity(),
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=None,
    )
    assert str(p).startswith(fake_root)


# ---------------------------------------------------------------------------
# Override args for non-standard sae_id strings
# ---------------------------------------------------------------------------


def test_default_target_cache_dir_override_activation_label_skips_parse(tmp_path):
    """When activation_label is explicitly set, _parse_sae_id is bypassed."""
    identity = _make_identity(sae_id="custom-non-parseable-id")
    p = default_target_cache_dir(
        suite="chat_target",
        sae_identity=identity,
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=tmp_path,
        activation_label="res",
        sae_width=16384,
    )
    assert "a_res" in p.parts
    assert "f16384" in p.parts


def test_default_target_cache_dir_override_width_only(tmp_path):
    """Both override args must be passed together — passing only one raises."""
    identity = _make_identity(sae_id="custom-non-parseable-id")
    with pytest.raises(ValueError, match="both activation_label and sae_width"):
        default_target_cache_dir(
            suite="chat_target",
            sae_identity=identity,
            model_max_seq_len=_CONTEXT_LEN,
            base_dir=tmp_path,
            activation_label="res",
            # sae_width missing
        )


def test_default_target_cache_dir_unparseable_sae_id_no_overrides_raises(tmp_path):
    identity = _make_identity(sae_id="totally-unknown-format")
    with pytest.raises(ValueError):
        default_target_cache_dir(
            suite="chat_target",
            sae_identity=identity,
            model_max_seq_len=_CONTEXT_LEN,
            base_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Hash truncation + sentinel
# ---------------------------------------------------------------------------


def test_default_target_cache_dir_truncates_hash_to_16_chars(tmp_path):
    identity = _make_identity(feature_selection_hash="0123456789abcdef" + "f" * 48)
    p = default_target_cache_dir(
        suite="chat_target",
        sae_identity=identity,
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=tmp_path,
    )
    # Last component must be exactly the first 16 hex chars.
    assert p.parts[-1] == "0123456789abcdef"
    assert len(p.parts[-1]) == HASH_TRUNCATION_LEN


def test_default_target_cache_dir_rejects_missing_hash():
    with pytest.raises(ValueError, match="feature_selection_hash is required"):
        _make_identity(feature_selection_hash=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Absolute paths invariant
# ---------------------------------------------------------------------------


def test_default_target_cache_dir_returns_absolute_path_when_base_absolute(tmp_path):
    p = default_target_cache_dir(
        suite="chat_target",
        sae_identity=_make_identity(),
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=tmp_path,
    )
    assert p.is_absolute(), f"expected absolute path; got {p}"


def test_default_target_cache_dir_accepts_string_base_dir(tmp_path):
    p = default_target_cache_dir(
        suite="chat_target",
        sae_identity=_make_identity(),
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=str(tmp_path),
    )
    assert isinstance(p, Path)


def test_default_target_cache_dir_with_different_pooling_modes(tmp_path):
    for pooling in ("max_pool", "mean_pool", "last_token"):
        identity = CacheIdentity(
            model_id="gemma-2-2b",
            sae_id="gemmascope-residual-l8-16k",
            layer=8,
            pooling=pooling,
            feature_selection_hash="a" * 64,
        )
        p = default_target_cache_dir(
            suite="chat_target",
            sae_identity=identity,
            model_max_seq_len=_CONTEXT_LEN,
            base_dir=tmp_path,
        )
        assert f"p_{pooling}" in p.parts


def test_default_target_cache_dir_invalid_pooling_raises(tmp_path):
    identity = CacheIdentity(
        model_id="gemma-2-2b",
        sae_id="gemmascope-residual-l8-16k",
        layer=8,
        pooling="bogus-pool",
        feature_selection_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="pooling"):
        default_target_cache_dir(
            suite="chat_target",
            sae_identity=identity,
            model_max_seq_len=_CONTEXT_LEN,
            base_dir=tmp_path,
        )


@_PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(max_value=0),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.none(),
    )
)
def test_default_target_cache_dir_rejects_every_invalid_context_length(
    invalid_length: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="model_max_seq_len"):
        default_target_cache_dir(
            suite="chat_target",
            sae_identity=_make_identity(),
            model_max_seq_len=cast(Any, invalid_length),
            base_dir=Path("/tmp/eval-cache-context-length-property"),
        )


def test_invalid_context_length_fails_before_sae_identity_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_calls = 0

    def forbidden_sae_parse(_sae_id: str) -> None:
        nonlocal parse_calls
        parse_calls += 1
        raise AssertionError("invalid context length must fail before SAE parsing")

    monkeypatch.setattr(eval_cache_paths, "_parse_sae_id", forbidden_sae_parse)

    with pytest.raises(BeartypeCallHintParamViolation, match="model_max_seq_len"):
        default_target_cache_dir(
            suite="chat_target",
            sae_identity=_make_identity(),
            model_max_seq_len=cast(Any, 0),
            base_dir=Path("/tmp/eval-cache-context-length-entry-order"),
        )

    assert parse_calls == 0


def test_default_target_cache_dir_empty_suite_raises(tmp_path):
    with pytest.raises(ValueError, match="suite"):
        default_target_cache_dir(
            suite="",
            sae_identity=_make_identity(),
            model_max_seq_len=_CONTEXT_LEN,
            base_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Component nesting matches feature_paths convention
# ---------------------------------------------------------------------------


def test_default_target_cache_dir_component_order(tmp_path):
    """Verify the path component order matches feature_paths convention:
    {base}/eval_caches/{suite}/m_{model}/l{layer}/a_{act}/p_{pool}/f{width}/c{T}/{hash16}
    """
    identity = _make_identity()
    p = default_target_cache_dir(
        suite="chat_target",
        sae_identity=identity,
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=tmp_path,
    )
    rel = p.relative_to(tmp_path).parts
    # Expected component order (after base_dir).
    assert rel[0] == "eval_caches"
    assert rel[1] == "chat_target"
    assert rel[2] == "m_gemma-2-2b"
    assert rel[3] == "l8"
    assert rel[4] == "a_res"
    assert rel[5] == "p_max_pool"
    assert rel[6] == "f16384"
    assert rel[7] == "c1024"
    assert rel[8] == "a" * HASH_TRUNCATION_LEN


def test_default_target_cache_dir_huggingface_model_name_sanitized(tmp_path):
    """HuggingFace-style model names must be sanitized via sanitize_model_name
    so the m_ component stays a single filesystem segment."""
    identity = CacheIdentity(
        model_id="google/gemma-2-2b",
        sae_id="gemmascope-residual-l8-16k",
        layer=8,
        pooling="max_pool",
        feature_selection_hash="a" * 64,
    )
    p = default_target_cache_dir(
        suite="chat_target",
        sae_identity=identity,
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=tmp_path,
    )
    # The "/" in the model name must be encoded as "_S" so m_google_Sgemma-2-2b
    # stays a single component.
    rel = p.relative_to(tmp_path).parts
    # Find the m_ component
    m_components = [c for c in rel if c.startswith("m_")]
    assert len(m_components) == 1
    assert "/" not in m_components[0]
    assert "_S" in m_components[0]
