"""Unit tests for ``steering._common.build_meta.build_gpt_meta``.

The helper is the single canonical implementation for shape-only GPT
construction across slice 11's training scripts (slice 11 audit /
consolidation manifest section 1.9). It must:

* produce a model whose tensors all live on the meta device,
* round ``depth * aspect_ratio`` UP to the nearest multiple of
  ``head_dim`` (including fractional ratios),
* derive ``n_head`` from the rounded model dim (``n_embd // head_dim``),
* equate ``n_kv_head`` with ``n_head`` (no GQA in the helper).
"""

from __future__ import annotations

import pytest
import torch

from steering._common.build_meta import build_gpt_meta, resolve_gpt_dimensions


@pytest.mark.parametrize("depth", [4, 6, 12, 13, 14, 26, 28])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_dim_rounding_to_head_dim_multiple(depth: int, head_dim: int) -> None:
    aspect_ratio = 64
    model = build_gpt_meta(
        depth=depth,
        aspect_ratio=aspect_ratio,
        head_dim=head_dim,
        max_seq_len=512,
        window_pattern="L",
        vocab_size=2048,
    )
    base_dim = depth * aspect_ratio
    expected_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    assert model.config.n_embd == expected_dim
    # Rounded dim must be a multiple of head_dim.
    assert model.config.n_embd % head_dim == 0
    assert model.config.n_head == expected_dim // head_dim


def test_n_kv_head_equals_n_head() -> None:
    model = build_gpt_meta(
        depth=12,
        aspect_ratio=64,
        head_dim=128,
        max_seq_len=512,
        window_pattern="L",
        vocab_size=2048,
    )
    assert model.config.n_kv_head == model.config.n_head


def test_all_params_on_meta_device() -> None:
    model = build_gpt_meta(
        depth=4,
        aspect_ratio=64,
        head_dim=64,
        max_seq_len=128,
        window_pattern="L",
        vocab_size=1024,
    )
    for param in model.parameters():
        assert param.device == torch.device("meta")
    for buf in model.buffers():
        assert buf.device == torch.device("meta")


def test_window_pattern_propagates() -> None:
    model = build_gpt_meta(
        depth=4,
        aspect_ratio=64,
        head_dim=64,
        max_seq_len=128,
        window_pattern="SSSL",
        vocab_size=1024,
    )
    assert model.config.window_pattern == "SSSL"


def test_seq_len_and_vocab_propagate() -> None:
    model = build_gpt_meta(
        depth=4,
        aspect_ratio=64,
        head_dim=64,
        max_seq_len=2048,
        window_pattern="L",
        vocab_size=32768,
    )
    assert model.config.sequence_len == 2048
    assert model.config.vocab_size == 32768


def test_rounding_up_when_base_dim_not_multiple() -> None:
    """depth=13, aspect_ratio=64, head_dim=128 -> 832 is not a multiple
    of 128; expect 896 (832 rounded up to next multiple of 128)."""
    model = build_gpt_meta(
        depth=13,
        aspect_ratio=64,
        head_dim=128,
        max_seq_len=512,
        window_pattern="L",
        vocab_size=2048,
    )
    assert model.config.n_embd == 896
    assert model.config.n_head == 7


def test_fractional_aspect_ratio_resolves_integer_target_width() -> None:
    """A fixed target width may be expressed as ``width / depth``."""
    model_dim, num_heads = resolve_gpt_dimensions(
        depth=27,
        aspect_ratio=1664 / 27,
        head_dim=128,
    )
    assert model_dim == 1664
    assert num_heads == 13
    assert type(model_dim) is int
    assert type(num_heads) is int

    model = build_gpt_meta(
        depth=27,
        aspect_ratio=1664 / 27,
        head_dim=128,
        max_seq_len=1024,
        window_pattern="L",
        vocab_size=2048,
    )
    assert model.config.n_embd == 1664
    assert model.config.n_head == 13
    assert model.config.n_kv_head == 13
    assert type(model.config.n_embd) is int
    assert type(model.config.n_head) is int
