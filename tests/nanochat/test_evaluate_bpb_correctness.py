"""Numeric-correctness tests for ``nanochat.loss_eval.evaluate_bpb``.

``evaluate_bpb`` converts summed per-token loss (nats) into bits-per-byte:
``bpb = total_nats / (ln(2) * total_bytes)`` where ``total_bytes`` is the sum of
the byte lengths of the *counted* target tokens.  The three exclusion rules that
make the metric vocab-independent are exercised explicitly:

1. "normal" tokens normalize by their byte length,
2. special tokens (``token_bytes == 0``) are masked out,
3. ignore_index targets (negative ``y``) contribute nothing.

A standalone mock model returns fixed per-token loss grids so the arithmetic is
fully hand-checkable.  nanochat is vendored/read-only — these tests only import
it, never modify it.
"""

import math

import torch
import torch.nn as nn

from nanochat.loss_eval import evaluate_bpb


class MockDecoderModel(nn.Module):
    """Returns fixed per-token loss grids for ``evaluate_bpb``.

    Each call consumes the next (B, T) tensor from ``loss_grid``; ``forward``
    asserts that ``evaluate_bpb`` requests ``loss_reduction="none"``.
    """

    def __init__(self, loss_grid: list[torch.Tensor], device: torch.device) -> None:
        super().__init__()
        self._loss_grid = list(loss_grid)
        self._call_idx = 0
        self._device = device
        # A real parameter keeps the module honest (non-empty .parameters()).
        self._dummy = nn.Parameter(torch.zeros(1, device=device))

    def get_device(self) -> torch.device:
        return self._device

    def forward(self, x, y, loss_reduction="mean"):
        assert loss_reduction == "none", (
            f"evaluate_bpb must request loss_reduction='none', got {loss_reduction!r}"
        )
        out = self._loss_grid[self._call_idx].to(self._device)
        self._call_idx += 1
        return out


def test_hand_computed_bpb() -> None:
    """Two tokens with known losses and byte counts.

    token_bytes: id 0 -> 3 bytes, id 1 -> 2 bytes; targets [0, 1] => bytes = 5.
    losses [1.5, 2.0] => nats = 3.5.  Expected bpb = 3.5 / (ln2 * 5).
    """
    device = torch.device("cpu")
    token_bytes = torch.tensor([3, 2], dtype=torch.int64, device=device)
    x = torch.tensor([[0, 1]], dtype=torch.long, device=device)
    y = torch.tensor([[0, 1]], dtype=torch.long, device=device)
    loss = torch.tensor([[1.5, 2.0]], dtype=torch.float32)

    model = MockDecoderModel(loss_grid=[loss], device=device)
    bpb = evaluate_bpb(model, [(x, y)], steps=1, token_bytes=token_bytes)

    expected = 3.5 / (math.log(2) * 5)
    assert math.isclose(bpb, expected, rel_tol=1e-6), f"expected {expected}, got {bpb}"


def test_special_tokens_excluded() -> None:
    """Tokens with ``token_bytes == 0`` contribute nothing to nats or bytes.

    token_bytes [3, 0, 2] (id 1 is special); targets [0, 1, 2];
    losses [1.0, 5.0, 2.0].  Only ids 0 and 2 count:
    nats = 1.0 + 2.0 = 3.0, bytes = 5, bpb = 3.0 / (ln2 * 5).
    """
    device = torch.device("cpu")
    token_bytes = torch.tensor([3, 0, 2], dtype=torch.int64, device=device)
    x = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    y = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    loss = torch.tensor([[1.0, 5.0, 2.0]], dtype=torch.float32)

    model = MockDecoderModel(loss_grid=[loss], device=device)
    bpb = evaluate_bpb(model, [(x, y)], steps=1, token_bytes=token_bytes)

    expected = 3.0 / (math.log(2) * 5)
    assert math.isclose(bpb, expected, rel_tol=1e-6), f"expected {expected}, got {bpb}"


def test_padding_ignore_index_excluded() -> None:
    """Targets with value -1 (ignore_index) contribute 0 nats and 0 bytes.

    token_bytes [4, 2]; targets [0, -1]; losses [2.0, 99.0] (99 ignored).
    nats = 2.0, bytes = 4, bpb = 2.0 / (ln2 * 4).
    """
    device = torch.device("cpu")
    token_bytes = torch.tensor([4, 2], dtype=torch.int64, device=device)
    x = torch.tensor([[0, 1]], dtype=torch.long, device=device)
    y = torch.tensor([[0, -1]], dtype=torch.long, device=device)
    loss = torch.tensor([[2.0, 99.0]], dtype=torch.float32)

    model = MockDecoderModel(loss_grid=[loss], device=device)
    bpb = evaluate_bpb(model, [(x, y)], steps=1, token_bytes=token_bytes)

    expected = 2.0 / (math.log(2) * 4)
    assert math.isclose(bpb, expected, rel_tol=1e-6), f"expected {expected}, got {bpb}"


def test_ignored_token_loss_does_not_leak() -> None:
    """A huge loss on an ignored (negative) target must not affect bpb.

    Identical setup to a single valid token, but with an extra ignore_index
    target carrying a large loss; the result must equal the single-token bpb.
    """
    device = torch.device("cpu")
    token_bytes = torch.tensor([4, 7], dtype=torch.int64, device=device)
    x = torch.tensor([[0, 1]], dtype=torch.long, device=device)

    y_valid_only = torch.tensor([[0, -1]], dtype=torch.long, device=device)
    loss = torch.tensor([[2.0, 1000.0]], dtype=torch.float32)
    model = MockDecoderModel(loss_grid=[loss], device=device)
    bpb_with_ignored = evaluate_bpb(
        model, [(x, y_valid_only)], steps=1, token_bytes=token_bytes
    )

    expected = 2.0 / (math.log(2) * 4)
    assert math.isclose(bpb_with_ignored, expected, rel_tol=1e-6)


def test_multi_batch_accumulation() -> None:
    """bpb accumulates summed nats/bytes across batches (it is not averaged).

    Batch 1: target [0], loss [1.0], bytes[0]=3 -> nats 1.0, bytes 3.
    Batch 2: target [1], loss [2.0], bytes[1]=2 -> nats 2.0, bytes 2.
    Accumulated nats 3.0, bytes 5 => bpb = 3.0 / (ln2 * 5).
    """
    device = torch.device("cpu")
    token_bytes = torch.tensor([3, 2], dtype=torch.int64, device=device)
    x1 = torch.tensor([[0]], dtype=torch.long, device=device)
    y1 = torch.tensor([[0]], dtype=torch.long, device=device)
    loss1 = torch.tensor([[1.0]], dtype=torch.float32)
    x2 = torch.tensor([[1]], dtype=torch.long, device=device)
    y2 = torch.tensor([[1]], dtype=torch.long, device=device)
    loss2 = torch.tensor([[2.0]], dtype=torch.float32)

    model = MockDecoderModel(loss_grid=[loss1, loss2], device=device)
    bpb = evaluate_bpb(model, [(x1, y1), (x2, y2)], steps=2, token_bytes=token_bytes)

    expected = 3.0 / (math.log(2) * 5)
    assert math.isclose(bpb, expected, rel_tol=1e-6), f"expected {expected}, got {bpb}"


def test_steps_limits_batches_consumed() -> None:
    """Only ``steps`` batches are consumed even if more are available.

    With steps=1, only the first batch (target [0], loss 1.0, bytes 3) counts:
    bpb = 1.0 / (ln2 * 3).  The second batch's large loss must be ignored.
    """
    device = torch.device("cpu")
    token_bytes = torch.tensor([3, 2], dtype=torch.int64, device=device)
    x1 = torch.tensor([[0]], dtype=torch.long, device=device)
    y1 = torch.tensor([[0]], dtype=torch.long, device=device)
    loss1 = torch.tensor([[1.0]], dtype=torch.float32)
    x2 = torch.tensor([[1]], dtype=torch.long, device=device)
    y2 = torch.tensor([[1]], dtype=torch.long, device=device)
    loss2 = torch.tensor([[500.0]], dtype=torch.float32)

    model = MockDecoderModel(loss_grid=[loss1, loss2], device=device)
    bpb = evaluate_bpb(model, [(x1, y1), (x2, y2)], steps=1, token_bytes=token_bytes)

    expected = 1.0 / (math.log(2) * 3)
    assert math.isclose(bpb, expected, rel_tol=1e-6), f"expected {expected}, got {bpb}"
    # Confirms steps governed consumption: only one forward happened.
    assert model._call_idx == 1


def test_all_zero_bytes_returns_inf() -> None:
    """If every counted token has 0 bytes, total_bytes==0 and bpb is +inf."""
    device = torch.device("cpu")
    token_bytes = torch.tensor([0, 0], dtype=torch.int64, device=device)
    x = torch.tensor([[0, 1]], dtype=torch.long, device=device)
    y = torch.tensor([[0, 1]], dtype=torch.long, device=device)
    loss = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    model = MockDecoderModel(loss_grid=[loss], device=device)
    bpb = evaluate_bpb(model, [(x, y)], steps=1, token_bytes=token_bytes)
    assert bpb == float("inf")


def test_all_ignored_targets_returns_inf() -> None:
    """If all targets are ignore_index, no bytes are counted -> +inf."""
    device = torch.device("cpu")
    token_bytes = torch.tensor([3, 2], dtype=torch.int64, device=device)
    x = torch.tensor([[0, 1]], dtype=torch.long, device=device)
    y = torch.tensor([[-1, -1]], dtype=torch.long, device=device)
    loss = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    model = MockDecoderModel(loss_grid=[loss], device=device)
    bpb = evaluate_bpb(model, [(x, y)], steps=1, token_bytes=token_bytes)
    assert bpb == float("inf")
