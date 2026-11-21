"""Unit tests for the residual-stream hook helper (Task 20).

Covers register_residual_hook from
steering.eval.axbench_base.baselines.hooks — the single shared helper
used by DiffMean, LinearProbe, and any future scalar-direction baseline
to add `coefficient * direction` to the residual output at a specific
transformer layer.

Edge cases pinned down here:
  * direction must be a 1-D (C,) tensor — 2-D/3-D raise AxBenchBaseError
  * layer out of range raises natural IndexError
  * coefficient == 0.0 is a bit-exact no-op
  * direction dtype is cast to output.dtype inside the hook so that
    the residual dtype of the model is preserved (bf16 stays bf16)
  * broadcasting works identically for prefill (B, T_prefill, C)
    and decode (B, 1, C)
  * device handling: direction is moved to the model device at
    registration time
  * returned handle is a torch.utils.hooks.RemovableHandle and
    .remove() restores the original forward path
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch as t
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from nanochat.gpt import GPT, GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.axbench_base.baselines.hooks import (
    register_batched_residual_hook,
    register_residual_hook,
)
from steering.eval.axbench_base.types import AxBenchBaseError


def _block(model: nn.Module, layer: int) -> nn.Module:
    """Fetch ``model.transformer.h[layer]`` as a typed ``nn.Module``.

    Mirrors the cast dance in
    ``steering.eval.axbench_base.baselines.hooks._get_block`` so the tests
    can type-check cleanly.
    """
    transformer = cast(nn.ModuleDict, model.transformer)
    blocks = cast(nn.ModuleList, transformer["h"])
    return blocks[layer]


# ---------------------------------------------------------------------------
# Mechanical fixtures — tiny models for fast CPU tests
# ---------------------------------------------------------------------------


TINY_VOCAB: int = 128
TINY_N_EMBD: int = 64
TINY_N_LAYER: int = 2
TINY_N_HEAD: int = 2
TINY_N_KV_HEAD: int = 2
TINY_SEQ_LEN: int = 128
TINY_HEAD_DIM: int = 32
TINY_N_SAE_FEATURES: int = 64
TINY_NUM_BUCKETS: int = 4


def _make_tiny_gpt_config() -> GPTConfig:
    return GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )


def _make_tiny_gpt() -> GPT:
    """Build a tiny GPT on CPU with fully initialised weights in eval mode."""
    config = _make_tiny_gpt_config()
    with t.device("meta"):
        model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


def _make_tiny_encdec() -> EncoderDecoderGPT:
    """Build a tiny EncoderDecoderGPT on CPU in eval mode."""
    dec_config = _make_tiny_gpt_config()
    enc_config = EncoderConfig(
        n_embd=TINY_N_EMBD,
        depth=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        head_dim=TINY_HEAD_DIM,
    )
    xattn_config = CrossAttentionConfig(frequency=1)
    model = EncoderDecoderGPT(
        dec_config=dec_config,
        enc_config=enc_config,
        xattn_config=xattn_config,
        n_sae_features=TINY_N_SAE_FEATURES,
        num_buckets=TINY_NUM_BUCKETS,
        pad_vocab_size_to=TINY_VOCAB,
    )
    model.init_weights()
    model.eval()
    return model


def _encdec_forward_kwargs(B: int, T: int) -> dict[str, Any]:
    """Return the extra kwargs EncoderDecoderGPT.forward needs for a
    simple prefill pass with one segment and a handful of features."""
    n_seg = 1
    num_features = 4
    idx = t.randint(0, TINY_VOCAB, (B, T))
    enc_feature_ids = t.randint(0, TINY_N_SAE_FEATURES, (B, n_seg, num_features))
    enc_magnitudes = t.randint(0, TINY_NUM_BUCKETS, (B, n_seg, num_features))
    enc_mask = t.ones(B, n_seg, num_features, dtype=t.bool)
    seg_map = t.zeros(B, T, dtype=t.long)
    return {
        "idx": idx,
        "targets": None,
        "enc_feature_ids": enc_feature_ids,
        "enc_magnitudes": enc_magnitudes,
        "enc_mask": enc_mask,
        "seg_map": seg_map,
    }


class _FakeBfloat16Block(nn.Module):
    """A deterministic bf16 block-shaped module.

    Used only by the dtype-mismatch test: building a real GPT in bf16
    would require CUDA (fp16/bf16 init is only applied on CUDA in
    GPT.init_weights), so we drop down to the smallest possible
    nn.Module stub that mirrors the `(B, T, C)` in→`(B, T, C)` out
    shape used by Block.forward.

    The helper only pokes at model.transformer.h[layer], so anything
    with that attribute path is accepted.
    """

    def __init__(self, C: int) -> None:
        super().__init__()
        self.C = C

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x * t.tensor(2.0, dtype=t.bfloat16)


class _FakeBfloat16Model(nn.Module):
    """Minimal model with the `transformer.h[i]` attribute path in bf16.

    The residual hook only needs this exact attribute path, per spec.
    """

    def __init__(self, C: int, n_layer: int) -> None:
        super().__init__()
        self.transformer = nn.ModuleDict(
            {"h": nn.ModuleList([_FakeBfloat16Block(C) for _ in range(n_layer)])}
        )
        # Single parameter pins the model's "device" for direction.to(...).
        self.device_anchor = nn.Parameter(t.zeros(1, dtype=t.bfloat16))


# ---------------------------------------------------------------------------
# Tests: basic additive semantics
# ---------------------------------------------------------------------------


class TestHookAdditive:
    """The hook should add `coefficient * direction` to the residual."""

    def test_hook_modifies_output_additively(self) -> None:
        model = _make_tiny_gpt()
        idx = t.randint(0, TINY_VOCAB, (1, 5))

        with t.no_grad():
            baseline = model(idx).clone()

        direction = t.randn(TINY_N_EMBD)
        coefficient = 0.75
        handle = register_residual_hook(model, layer=1, direction=direction,
                                         coefficient=coefficient)
        try:
            with t.no_grad():
                hooked = model(idx).clone()
        finally:
            handle.remove()

        # Hook adds to the Block output (residual) at layer 1, which then
        # passes through the final norm + lm_head. We cannot check the
        # final logits match the direction arithmetically, but they must
        # differ.
        assert not t.equal(baseline, hooked), (
            "Logits should differ when hook adds a nonzero direction."
        )

    def test_hook_modifies_block_output_by_exact_amount(self) -> None:
        """Capture Block layer-0's output by capturing layer-1's input
        (which equals ``resid_lambdas[1] * block0_out + x0_lambdas[1] * x0``).

        We compare the diff between hooked and unhooked captures:
        it must equal ``resid_lambdas[1] * coefficient * direction``
        at every (b, t) position, within the precision of COMPUTE_DTYPE
        (bf16 on ampere+, fp32 elsewhere).
        """
        model = _make_tiny_gpt()
        idx = t.randint(0, TINY_VOCAB, (2, 3))

        captures: list[t.Tensor] = []

        def capture_input(module, inputs):
            captures.append(inputs[0].detach().clone())

        cap_handle = _block(model, 1).register_forward_pre_hook(capture_input)
        try:
            with t.no_grad():
                model(idx)
            unhooked_resid = captures.pop()

            direction = t.randn(TINY_N_EMBD)
            coefficient = 0.5
            hook_handle = register_residual_hook(
                model, layer=0, direction=direction, coefficient=coefficient
            )
            try:
                with t.no_grad():
                    model(idx)
                hooked_resid = captures.pop()
            finally:
                hook_handle.remove()
        finally:
            cap_handle.remove()

        diff = (hooked_resid - unhooked_resid).to(t.float32)
        # Expected: resid_lambdas[1] * coefficient * direction, applied
        # identically to every (b, t) slot. x0_lambdas * x0 cancels on
        # subtraction because the initial-embedding branch does not
        # depend on the hook.
        lam = model.resid_lambdas[1].detach().to(t.float32)
        expected_per_slot = (lam * coefficient) * direction.to(t.float32)
        expected = expected_per_slot[None, None, :].expand_as(diff)
        # Tolerance large enough for bf16 round-off accumulated across
        # the resid_lambdas fp32 multiply + bf16 cast.
        assert t.allclose(diff, expected, atol=5e-2, rtol=5e-2), (
            "Residual modification is not proportional to coefficient * "
            "direction broadcast across (b, t)."
        )

    def test_direction_broadcasts_across_batch_and_time(self) -> None:
        """B=2, T=3: every (b, t) slot gets the same additive term.

        Uses the stub bf16 model so we can control the block output
        deterministically (stub returns ``x * 2``, so feeding a zero
        tensor gives a zero baseline and the hook contribution is
        isolated).
        """
        C = 8
        model = _FakeBfloat16Model(C=C, n_layer=2)
        x = t.zeros(2, 3, C, dtype=t.bfloat16)
        direction = t.arange(C, dtype=t.bfloat16)
        coefficient = 1.25

        handle = register_residual_hook(
            model, layer=0, direction=direction, coefficient=coefficient
        )
        try:
            with t.no_grad():
                hooked = _block(model, 0)(x)
        finally:
            handle.remove()

        # Baseline was all zeros so hooked == addend at every position.
        expected_row = (coefficient * direction).to(t.bfloat16)
        B, T, _ = hooked.shape
        for b in range(B):
            for ti in range(T):
                assert t.equal(hooked[b, ti, :], expected_row), (
                    f"Row (b={b}, t={ti}) does not match expected addend."
                )

# ---------------------------------------------------------------------------
# Tests: prefill + decode compatibility
# ---------------------------------------------------------------------------


class TestHookPrefillDecode:
    def test_hook_fires_during_both_prefill_and_decode(self) -> None:
        """Run one (1, 5) forward (prefill-shape) and one (1, 1) forward
        (decode-shape) — the hook must fire on both."""
        model = _make_tiny_gpt()
        direction = t.randn(TINY_N_EMBD)
        coefficient = 0.6

        prefill_idx = t.randint(0, TINY_VOCAB, (1, 5))
        decode_idx = t.randint(0, TINY_VOCAB, (1, 1))

        with t.no_grad():
            prefill_baseline = model(prefill_idx).clone()
            decode_baseline = model(decode_idx).clone()

        call_count = 0

        def counting_hook(module, inputs, output):
            nonlocal call_count
            call_count += 1
            return output

        # Install a counting hook first so we can count fires regardless
        # of what the real helper does.
        counter_handle = _block(model, 1).register_forward_hook(counting_hook)
        real_handle = register_residual_hook(
            model, layer=1, direction=direction, coefficient=coefficient
        )
        try:
            with t.no_grad():
                prefill_hooked = model(prefill_idx).clone()
                decode_hooked = model(decode_idx).clone()
        finally:
            real_handle.remove()
            counter_handle.remove()

        # Both calls should have registered modifications
        assert call_count == 2, f"Expected 2 hook fires, got {call_count}."
        assert not t.equal(prefill_baseline, prefill_hooked), (
            "Hook did not modify prefill output."
        )
        assert not t.equal(decode_baseline, decode_hooked), (
            "Hook did not modify decode output."
        )


# ---------------------------------------------------------------------------
# Tests: removal restores the original forward path
# ---------------------------------------------------------------------------


class TestHookRemoval:
    def test_hook_removed_restores_original_output(self) -> None:
        model = _make_tiny_gpt()
        idx = t.randint(0, TINY_VOCAB, (1, 4))

        with t.no_grad():
            baseline = model(idx).clone()

        direction = t.randn(TINY_N_EMBD)
        handle = register_residual_hook(
            model, layer=0, direction=direction, coefficient=0.9
        )
        with t.no_grad():
            hooked = model(idx).clone()
        assert not t.equal(baseline, hooked)

        handle.remove()

        with t.no_grad():
            restored = model(idx).clone()
        assert t.equal(baseline, restored), (
            "After .remove() the output should be bit-exact identical to baseline."
        )

    def test_returned_handle_is_removable_handle(self) -> None:
        model = _make_tiny_gpt()
        direction = t.randn(TINY_N_EMBD)
        handle = register_residual_hook(
            model, layer=0, direction=direction, coefficient=1.0
        )
        try:
            assert isinstance(handle, RemovableHandle), (
                f"Expected torch.utils.hooks.RemovableHandle, got {type(handle)}."
            )
        finally:
            handle.remove()


# ---------------------------------------------------------------------------
# Tests: works on both GPT and EncoderDecoderGPT (shared attribute path)
# ---------------------------------------------------------------------------


class TestModelCompatibility:
    def test_hook_works_on_gpt_model(self) -> None:
        """Already covered by TestHookAdditive; restated here to pin
        intent: the helper must work on a plain nanochat.gpt.GPT."""
        model = _make_tiny_gpt()
        idx = t.randint(0, TINY_VOCAB, (1, 3))
        direction = t.randn(TINY_N_EMBD)

        with t.no_grad():
            baseline = model(idx).clone()

        handle = register_residual_hook(
            model, layer=1, direction=direction, coefficient=1.0
        )
        try:
            with t.no_grad():
                hooked = model(idx).clone()
        finally:
            handle.remove()

        assert not t.equal(baseline, hooked)

    def test_hook_works_on_encdec_model(self) -> None:
        """Same helper, same attribute path, different class."""
        model = _make_tiny_encdec()
        kwargs = _encdec_forward_kwargs(B=1, T=4)

        with t.no_grad():
            baseline = model(**kwargs).clone()

        direction = t.randn(TINY_N_EMBD)
        handle = register_residual_hook(
            model, layer=1, direction=direction, coefficient=0.5
        )
        try:
            with t.no_grad():
                hooked = model(**kwargs).clone()
        finally:
            handle.remove()

        assert not t.equal(baseline, hooked)

        # And .remove() restores the original behaviour bit-exactly.
        with t.no_grad():
            restored = model(**kwargs).clone()
        assert t.equal(baseline, restored)


# ---------------------------------------------------------------------------
# Tests: coefficient == 0 is a bit-exact no-op
# ---------------------------------------------------------------------------


class TestCoefficientZero:
    def test_coefficient_zero_is_no_op(self) -> None:
        model = _make_tiny_gpt()
        idx = t.randint(0, TINY_VOCAB, (1, 6))

        with t.no_grad():
            baseline = model(idx).clone()

        direction = t.randn(TINY_N_EMBD)
        handle = register_residual_hook(
            model, layer=1, direction=direction, coefficient=0.0
        )
        try:
            with t.no_grad():
                hooked = model(idx).clone()
        finally:
            handle.remove()

        assert t.equal(baseline, hooked), (
            "coefficient=0.0 must be a bit-exact no-op."
        )


# ---------------------------------------------------------------------------
# Tests: layer out-of-range raises a natural IndexError
# ---------------------------------------------------------------------------


class TestLayerOutOfRange:
    @pytest.mark.parametrize("bad_layer", [TINY_N_LAYER, TINY_N_LAYER + 5, 999])
    def test_layer_out_of_range_raises_index_error(self, bad_layer: int) -> None:
        model = _make_tiny_gpt()
        direction = t.randn(TINY_N_EMBD)
        with pytest.raises(IndexError):
            register_residual_hook(
                model, layer=bad_layer, direction=direction, coefficient=1.0
            )


# ---------------------------------------------------------------------------
# Tests: direction shape validation
# ---------------------------------------------------------------------------


class TestDirectionShape:
    @pytest.mark.parametrize(
        "shape",
        [
            (1, TINY_N_EMBD),       # 2-D
            (TINY_N_EMBD, 1),       # 2-D other way
            (1, 1, TINY_N_EMBD),    # 3-D
            (),                     # 0-D scalar
        ],
    )
    def test_direction_wrong_ndim_raises(self, shape: tuple[int, ...]) -> None:
        model = _make_tiny_gpt()
        direction = t.randn(*shape) if shape else t.tensor(1.0)
        with pytest.raises(AxBenchBaseError, match="direction"):
            register_residual_hook(
                model, layer=0, direction=direction, coefficient=1.0
            )

    def test_direction_wrong_size_is_caller_responsibility(self) -> None:
        """A (C,) tensor whose C does not match the model's n_embd is the
        caller's problem — broadcasting will fail naturally inside the hook.

        We document that failure mode here so a future refactor cannot
        silently hide it.
        """
        model = _make_tiny_gpt()
        direction = t.randn(TINY_N_EMBD + 1)  # wrong size but right ndim
        idx = t.randint(0, TINY_VOCAB, (1, 3))

        handle = register_residual_hook(
            model, layer=0, direction=direction, coefficient=1.0
        )
        try:
            with pytest.raises(RuntimeError):
                with t.no_grad():
                    model(idx)
        finally:
            handle.remove()


# ---------------------------------------------------------------------------
# Tests: device handling
# ---------------------------------------------------------------------------


class TestDeviceHandling:
    def test_direction_on_model_device_works(self) -> None:
        """CPU-only smoke test: pass a CPU direction to a CPU model."""
        model = _make_tiny_gpt()
        direction = t.randn(TINY_N_EMBD, device="cpu")
        idx = t.randint(0, TINY_VOCAB, (1, 3))

        handle = register_residual_hook(
            model, layer=0, direction=direction, coefficient=0.4
        )
        try:
            with t.no_grad():
                out = model(idx)
            assert out.shape == (1, 3, TINY_VOCAB)
        finally:
            handle.remove()

    @pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
    def test_direction_on_different_device_moved_to_model_device(self) -> None:
        """Register with a CPU direction against a CUDA model: the helper
        moves the direction to the model's device so broadcasting works.
        """
        config = _make_tiny_gpt_config()
        with t.device("meta"):
            model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
        model = model.to_empty(device="cuda")
        model.init_weights()
        model.eval()

        direction_cpu = t.randn(TINY_N_EMBD, device="cpu")
        idx = t.randint(0, TINY_VOCAB, (1, 3), device="cuda")

        handle = register_residual_hook(
            model, layer=0, direction=direction_cpu, coefficient=0.4
        )
        try:
            with t.no_grad():
                out = model(idx)
            assert out.device.type == "cuda"
        finally:
            handle.remove()


# ---------------------------------------------------------------------------
# Tests: dtype handling
# ---------------------------------------------------------------------------


class TestDtypeHandling:
    def test_fp32_direction_against_real_gpt_preserves_block_dtype(self) -> None:
        """Fp32 direction into a real GPT: the block output dtype
        (COMPUTE_DTYPE — bf16 on Ampere+, fp32 on CPU-only) must be
        preserved after the hook adds the direction.

        We capture the output of the hooked block directly via a second
        forward hook that records ``output.dtype``.
        """
        model = _make_tiny_gpt()
        idx = t.randint(0, TINY_VOCAB, (1, 3))

        captured_dtypes: list[t.dtype] = []

        def record_dtype(
            module: nn.Module,
            inputs: tuple[t.Tensor, ...],
            output: t.Tensor,
        ) -> None:
            del module, inputs
            captured_dtypes.append(output.dtype)

        # First: baseline block dtype.
        dtype_handle = _block(model, 0).register_forward_hook(record_dtype)
        try:
            with t.no_grad():
                model(idx)
        finally:
            dtype_handle.remove()
        baseline_dtype = captured_dtypes[-1]

        # Now install the residual hook (adds fp32 direction) and re-record.
        direction = t.randn(TINY_N_EMBD, dtype=t.float32)
        hook_handle = register_residual_hook(
            model, layer=0, direction=direction, coefficient=0.4
        )
        # Re-install the dtype recorder AFTER the residual hook so it sees
        # the hook's output (forward hooks fire in registration order).
        dtype_handle = _block(model, 0).register_forward_hook(record_dtype)
        try:
            with t.no_grad():
                model(idx)
        finally:
            dtype_handle.remove()
            hook_handle.remove()
        hooked_dtype = captured_dtypes[-1]

        assert hooked_dtype == baseline_dtype, (
            f"Hook changed block output dtype: baseline={baseline_dtype}, "
            f"hooked={hooked_dtype}."
        )

    def test_fp32_direction_against_bf16_block_preserves_bf16(self) -> None:
        """Pin edge case #4: we decided to cast `direction` to `output.dtype`
        inside the hook, so the residual dtype is preserved. A fp32
        direction against a bf16 block output must NOT upcast the
        downstream residual to fp32.
        """
        model = _FakeBfloat16Model(C=8, n_layer=2)
        x = t.ones(1, 3, 8, dtype=t.bfloat16)

        # Unhooked: stub block returns x * 2.0
        with t.no_grad():
            baseline = _block(model, 0)(x)
        assert baseline.dtype == t.bfloat16

        direction = t.arange(8, dtype=t.float32)  # fp32 on purpose
        handle = register_residual_hook(
            model, layer=0, direction=direction, coefficient=0.5
        )
        try:
            with t.no_grad():
                hooked = _block(model, 0)(x)
        finally:
            handle.remove()

        assert hooked.dtype == t.bfloat16, (
            f"Hook must preserve bf16 output dtype, got {hooked.dtype}."
        )
        # Numerical: hooked == baseline + 0.5 * direction (in bf16).
        expected = baseline + (0.5 * direction.to(t.bfloat16))[None, None, :]
        assert t.equal(hooked, expected), (
            "Numerical result does not match bf16(baseline) + "
            "bf16(coefficient * direction)."
        )


class TestBatchedResidualHook:
    def test_each_row_receives_only_its_own_addend_at_every_position(self) -> None:
        model = _FakeBfloat16Model(C=8, n_layer=2)
        x = t.arange(3 * 5 * 8, dtype=t.bfloat16).reshape(3, 5, 8)
        addends = t.tensor(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
                [0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5],
            ],
            dtype=t.float32,
        )
        coefficients = (0.25, -1.5, 2.0)
        with t.no_grad():
            baseline = _block(model, 1)(x)
        handle = register_batched_residual_hook(model, 1, addends, coefficients)
        try:
            with t.no_grad():
                actual = _block(model, 1)(x)
        finally:
            handle.remove()

        scaled = t.stack(
            [
                coefficient * direction.to(t.bfloat16)
                for coefficient, direction in zip(coefficients, addends, strict=True)
            ]
        )
        expected = baseline + scaled[:, None, :]
        assert actual.dtype == t.bfloat16
        assert t.equal(actual, expected)
        for row in range(3):
            assert t.equal(actual[row] - baseline[row], expected[row] - baseline[row])

    def test_remove_restores_exact_output(self) -> None:
        model = _FakeBfloat16Model(C=8, n_layer=1)
        x = t.randn(2, 1, 8, dtype=t.bfloat16)
        with t.no_grad():
            baseline = _block(model, 0)(x)
        handle = register_batched_residual_hook(model, 0, t.ones(2, 8), (1.0, 1.0))
        with t.no_grad():
            assert not t.equal(_block(model, 0)(x), baseline)
        handle.remove()
        with t.no_grad():
            assert t.equal(_block(model, 0)(x), baseline)

    @pytest.mark.parametrize("shape", [(8,), (1, 2, 8), (0, 8), (2, 0)])
    def test_invalid_addend_shape_fails_at_registration(self, shape: tuple[int, ...]) -> None:
        model = _FakeBfloat16Model(C=8, n_layer=1)
        with pytest.raises(AxBenchBaseError, match="batched residual directions"):
            register_batched_residual_hook(model, 0, t.empty(shape), (1.0,))

    def test_coefficient_cardinality_mismatch_fails_at_registration(self) -> None:
        model = _FakeBfloat16Model(C=8, n_layer=1)
        with pytest.raises(AxBenchBaseError, match="one coefficient per direction"):
            register_batched_residual_hook(model, 0, t.ones(2, 8), (1.0,))

    @pytest.mark.parametrize(
        ("x_shape", "message"),
        [
            ((2, 8), "expected block output"),
            ((3, 1, 8), "disagrees with directions"),
            ((2, 1, 7), "disagrees with directions"),
        ],
    )
    def test_live_output_mismatch_fails_without_broadcast(
        self,
        x_shape: tuple[int, ...],
        message: str,
    ) -> None:
        model = _FakeBfloat16Model(C=x_shape[-1], n_layer=1)
        handle = register_batched_residual_hook(model, 0, t.ones(2, 8), (1.0, 1.0))
        try:
            with pytest.raises(AxBenchBaseError, match=message):
                _block(model, 0)(t.zeros(x_shape, dtype=t.bfloat16))
        finally:
            handle.remove()
