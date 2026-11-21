"""Verify torch.compile(dynamic=True) produces identical results to uncompiled model.

These tests require CUDA (torch.compile is not meaningful on CPU).
"""

import pytest
import torch
from torch._dynamo.utils import counters

from nanochat.gpt import GPTConfig
from steering._common.compile_helpers import resolve_compile_dynamic
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for torch.compile tests"
)


def _build_model(device="cuda"):
    """Build a small EncoderDecoderGPT for testing."""
    dec_config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
    )
    enc_config = EncoderConfig(n_embd=32, depth=2, n_head=2, n_kv_head=2, head_dim=16)
    xattn_config = CrossAttentionConfig(frequency=2)
    model = EncoderDecoderGPT(
        dec_config,
        enc_config,
        xattn_config,
        n_sae_features=128,
        num_buckets=10,
    ).to(device)
    model.init_weights()
    return model


def _make_inputs(model, B=2, T=32, n_seg=2, F=8, device="cuda"):
    """Create random inputs for the model."""
    return {
        "idx": torch.randint(0, model.dec_config.vocab_size, (B, T), device=device),
        "targets": torch.randint(0, model.dec_config.vocab_size, (B, T), device=device),
        "enc_feature_ids": torch.randint(0, 128, (B, n_seg, F), device=device),
        "enc_magnitudes": torch.randint(0, 10, (B, n_seg, F), device=device),
        "enc_mask": torch.ones(B, n_seg, F, dtype=torch.bool, device=device),
        "seg_map": torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        // (T // n_seg),
    }


class TestCompileCorrectness:
    """Verify compiled model matches uncompiled model."""

    def test_loss_matches(self):
        """Compiled and uncompiled forward produce the same loss."""
        torch.manual_seed(42)
        model = _build_model()
        inputs = _make_inputs(model)

        # Uncompiled reference
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ref_loss = model(**inputs)

        # Compiled
        compiled_model = torch.compile(model, dynamic=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            compiled_loss = compiled_model(**inputs)

        rtol = abs(compiled_loss.item() - ref_loss.item()) / abs(ref_loss.item())
        assert rtol < 1e-4, (
            f"Loss mismatch: ref={ref_loss.item()}, compiled={compiled_loss.item()}, rtol={rtol}"
        )

    def test_gradient_norms_match(self):
        """Compiled and uncompiled backward produce similar gradient norms."""
        torch.manual_seed(42)
        model_ref = _build_model()
        model_compiled = _build_model()
        # Sync weights
        model_compiled.load_state_dict(model_ref.state_dict())

        inputs = _make_inputs(model_ref)

        # Uncompiled backward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_ref = model_ref(**inputs)
        loss_ref.backward()
        ref_grad_norm = torch.nn.utils.clip_grad_norm_(
            model_ref.parameters(), float("inf")
        )

        # Compiled backward
        compiled = torch.compile(model_compiled, dynamic=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_compiled = compiled(**inputs)
        loss_compiled.backward()
        compiled_grad_norm = torch.nn.utils.clip_grad_norm_(
            model_compiled.parameters(), float("inf")
        )

        rtol = abs(compiled_grad_norm.item() - ref_grad_norm.item()) / abs(
            ref_grad_norm.item()
        )
        assert rtol < 1e-2, (
            f"Gradient norm mismatch: ref={ref_grad_norm.item()}, "
            f"compiled={compiled_grad_norm.item()}, rtol={rtol}"
        )

    def test_varying_shapes(self):
        """Compiled model handles varying input shapes without recompilation errors."""
        model = _build_model()
        compiled_model = torch.compile(model, dynamic=True)

        # Run with different shapes
        for T, n_seg, F in [(32, 2, 8), (16, 1, 4), (48, 3, 12)]:
            inputs = _make_inputs(model, B=2, T=T, n_seg=n_seg, F=F)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = compiled_model(**inputs)
            assert not torch.isnan(loss), f"NaN loss for T={T}, n_seg={n_seg}, F={F}"

    def test_static_outer_graph_reuses_dynamic_packed_subgraphs(self):
        """Packed subgraphs generalize once, then reuse across new lengths."""
        torch.manual_seed(20260817)
        model = _build_model()
        compiled_model = torch.compile(model, dynamic=False)
        inputs = _make_inputs(model, B=2, T=32, n_seg=2, F=100)
        flat_positions = torch.arange(400, device="cuda").reshape(2, 2, 100)
        active_counts = (161, 225, 289, 353, 161)

        torch._dynamo.reset()
        counters.clear()
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                inputs["enc_mask"] = flat_positions < active_counts[0]
                compiled_model(**inputs)
                # A second ordinary warmup resolves nanochat's existing
                # value-embedding/window variants before measuring K guards.
                compiled_model(**inputs)
            torch.cuda.synchronize()
            first_graph_count = counters["stats"]["unique_graphs"]
            assert first_graph_count > 0

            for active_count in active_counts[1:]:
                inputs["enc_mask"] = flat_positions < active_count
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = compiled_model(**inputs)
                assert torch.isfinite(loss)
            torch.cuda.synchronize()
            assert counters["stats"]["unique_graphs"] == first_graph_count
        finally:
            torch._dynamo.reset()


class TestResolvedSFTPolicyCorrectness:
    """Exercise EncoderDecoderGPT under the RESOLVED SFT compile policy.

    ``resolve_compile_dynamic("auto", mode="sft")`` returns ``None``
    (auto-dynamic): decoder dims stay concrete — required by the torch>=2.13
    flex FLASH (CuteDSL) backend on Hopper, which cannot inline a symbolic
    mask_mod capture — while the ``mark_dynamic``'d packed tensors and
    auto-generalized encoder input dims compile symbolically.  These tests
    pin loss parity and graph stability for that policy on the real model
    (any CUDA arch; the FLASH-specific crash itself only reproduces on
    Hopper+, where the SFT run is the end-to-end gate).
    """

    def test_loss_matches_eager_under_resolved_policy(self):
        torch.manual_seed(42)
        model = _build_model()
        inputs = _make_inputs(model)

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ref_loss = model(**inputs)

        dynamic = resolve_compile_dynamic("auto", mode="sft")
        assert dynamic is None
        # Production warms the decoder BlockMask cache before compiling
        # (conditional_encdec_sft.py); mirror it with the batch geometry the
        # inputs actually use so the trace never rebuilds a mask in-graph —
        # on Hopper/FA4 a cold-cache rebuild is the crash under test.
        model.prewarm_decoder_block_masks(2, 32)
        compiled_model = torch.compile(model, dynamic=dynamic)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            compiled_loss = compiled_model(**inputs)

        rtol = abs(compiled_loss.item() - ref_loss.item()) / abs(ref_loss.item())
        assert rtol < 1e-4, (
            f"Loss mismatch: ref={ref_loss.item()}, compiled={compiled_loss.item()}, rtol={rtol}"
        )

    def test_varying_packed_shapes_stabilize_under_resolved_policy(self):
        """Fixed decoder (B, T) + varying (n_seg, F): graphs must plateau.

        Production SFT keeps decoder rows a fixed shape and varies only the
        conditioning extents per batch.  Under auto-dynamic each varying dim
        generalizes on its first change; afterwards both repeats and NEW
        extents must reuse the existing graphs (no per-batch recompiles —
        the property the old ``dynamic=True`` bought, preserved by ``None``).

        Extents mirror the production regime: packed K = B*n_seg*F stays
        >= 256 so every BlockMask sparse extent exceeds 1 from the first
        trace (``mark_packed_block_mask_dynamic`` skips size-1 dims — torch
        0/1-specializes those by design) and K never crosses back under the
        128-token tile boundary, which would legitimately bifurcate the
        packed graphs once.  Production packed counts are in the thousands,
        so neither small-size effect occurs there.
        """
        torch.manual_seed(20260825)
        model = _build_model()
        dynamic = resolve_compile_dynamic("auto", mode="sft")
        assert dynamic is None
        model.prewarm_decoder_block_masks(2, 32)  # mirror production (see above)
        compiled_model = torch.compile(model, dynamic=dynamic)

        torch._dynamo.reset()
        counters.clear()
        try:
            # Warmup: first change of each varying dim triggers its one
            # generalization recompile (plus nanochat's window variants).
            # n_seg must divide T=32 (seg_map construction in _make_inputs).
            for n_seg, feature_count in [(4, 32), (8, 48), (16, 64)]:
                inputs = _make_inputs(model, B=2, T=32, n_seg=n_seg, F=feature_count)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = compiled_model(**inputs)
                assert torch.isfinite(loss)
            torch.cuda.synchronize()
            warm_graphs = counters["stats"]["unique_graphs"]
            assert warm_graphs > 0

            # Post-generalization: a repeat AND unseen extents add no graphs.
            for n_seg, feature_count in [(4, 32), (8, 40), (16, 56)]:
                inputs = _make_inputs(model, B=2, T=32, n_seg=n_seg, F=feature_count)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = compiled_model(**inputs)
                assert torch.isfinite(loss)
            torch.cuda.synchronize()
            assert counters["stats"]["unique_graphs"] == warm_graphs, (
                "post-generalization packed extents must not recompile: "
                f"{counters['stats']['unique_graphs']} != {warm_graphs}"
            )
        finally:
            torch._dynamo.reset()

    def test_exhausted_val_rank_padding_batch_transition(self):
        """Normal batches -> the exhausted-validation-rank payload -> back.

        A rank whose bounded validation slice runs out mid-pass emits
        all-padding rows with ``(B, 1, 1)`` encoder tensors and an all-False
        mask.  That shape crosses exactly the 0/1-specialization and 128-tile
        regimes the stability test above excludes, so it MAY add one graph
        family on first sight — and must then be stable: repeats of both the
        padded and normal shapes add nothing, and no forward may error under
        the resolved compile policy with the live attention backend (the
        existing training-loop CUDA test covers this payload only under a
        forced-SDPA, dynamic=False compile).
        """
        torch.manual_seed(20260826)
        model = _build_model()
        dynamic = resolve_compile_dynamic("auto", mode="sft")
        model.prewarm_decoder_block_masks(2, 32)  # mirror production (see above)
        compiled_model = torch.compile(model, dynamic=dynamic)

        def run_normal():
            inputs = _make_inputs(model, B=2, T=32, n_seg=4, F=32)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_flat = compiled_model(**inputs, loss_reduction="none")
            covered = inputs["targets"].reshape(-1) >= 0
            assert torch.isfinite(loss_flat[covered]).all()

        def run_padded():
            inputs = _exhausted_rank_inputs(model, B=2, T=32)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_flat = compiled_model(**inputs, loss_reduction="none")
            # Every target is -1: the payload contributes nothing downstream
            # (consumers mask by target), but the forward itself must survive.
            assert loss_flat.numel() == 2 * 32

        torch._dynamo.reset()
        counters.clear()
        try:
            run_normal()
            run_normal()
            torch.cuda.synchronize()
            before_padded = counters["stats"]["unique_graphs"]
            run_padded()  # first sight: a bounded new graph family is legitimate
            torch.cuda.synchronize()
            after_first_padded = counters["stats"]["unique_graphs"]
            assert after_first_padded - before_padded <= 4, (
                "first padded batch may add a small graph family (0/1 and "
                "128-tile regime change), not an explosion: "
                f"+{after_first_padded - before_padded}"
            )
            run_padded()
            run_normal()
            run_padded()
            torch.cuda.synchronize()
            assert counters["stats"]["unique_graphs"] == after_first_padded, (
                "padded/normal alternation must not keep recompiling: "
                f"{counters['stats']['unique_graphs']} != {after_first_padded}"
            )
        finally:
            torch._dynamo.reset()


def _exhausted_rank_inputs(model, B=2, T=32, device="cuda"):
    """The exhausted-validation-rank payload: all-padding rows, (B,1,1) encoder
    tensors with an all-False mask, and all ``-1`` targets (see the exhausted
    -rank contract in scripts/train/conditional_encdec_sft.py)."""
    return {
        "idx": torch.zeros(B, T, dtype=torch.int32, device=device),
        "targets": torch.full((B, T), -1, dtype=torch.int64, device=device),
        "enc_feature_ids": torch.zeros(B, 1, 1, dtype=torch.int64, device=device),
        "enc_magnitudes": torch.zeros(B, 1, 1, dtype=torch.int64, device=device),
        "enc_mask": torch.zeros(B, 1, 1, dtype=torch.bool, device=device),
        "seg_map": torch.zeros(B, T, dtype=torch.int64, device=device),
    }


def _flex_hop_other_buffers(graphs):
    """Collect (score_others, mask_others) from every flex_attention HOP."""
    collected = []
    for gm in graphs:
        for node in gm.graph.nodes:
            if node.op == "call_function" and "flex_attention" in str(node.target):
                score_others = node.args[7] if len(node.args) > 7 else ()
                mask_others = node.args[8] if len(node.args) > 8 else ()
                collected.append((score_others, mask_others))
    return collected


class TestDecoderFlexScalarCapture:
    """Pin the FLASH-backend lowering precondition on the traced graphs.

    torch>=2.13's flex FLASH (CuteDSL) backend refuses any symbolic scalar in
    a mask_mod's captured buffers ("NYI: score_mod or mask_mod captures a
    dynamic scalar").  That precondition — every flex HOP traces with EMPTY
    other_buffers — is backend-independent, so it is assertable on any CUDA
    arch even though the crash itself only reproduces on Hopper.  Two
    steering-side measures uphold it: ``_decoder_blocks`` avoids the
    untraceable ``typing.cast`` generic subscript (which hard-broke the
    decoder graph), and ``prewarm_decoder_block_masks`` ensures the traced
    ``_get_block_mask`` HITS nanochat's cache, so the mask_mod closes over a
    concrete Python int instead of an in-graph SymInt.
    """

    @staticmethod
    def _trace_forward(model, prewarm: bool):
        from nanochat import flash_attention as fa

        graphs = []

        def capture_backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        fa._block_mask_cache.clear()
        torch._dynamo.reset()
        try:
            if prewarm:
                model.prewarm_decoder_block_masks(2, model.dec_config.sequence_len)
            compiled = torch.compile(model, backend=capture_backend, dynamic=None)
            inputs = _make_inputs(
                model, B=2, T=model.dec_config.sequence_len, n_seg=4, F=32
            )
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                compiled(**inputs)
        finally:
            torch._dynamo.reset()
            fa._block_mask_cache.clear()
        return graphs

    def test_prewarmed_trace_has_no_flex_scalar_captures(self):
        from nanochat import flash_attention as fa

        if not fa._use_flex():
            pytest.skip("flex path inactive (sdpa/fa3 override or unsupported)")
        torch.manual_seed(7)
        model = _build_model()
        graphs = self._trace_forward(model, prewarm=True)
        buffers = _flex_hop_other_buffers(graphs)
        assert buffers, "expected at least one flex_attention HOP in the trace"
        for score_others, mask_others in buffers:
            assert score_others == () and mask_others == (), (
                "flex HOP captured extra buffers — the FLASH (CuteDSL) backend "
                f"would reject a symbolic scalar here: score={score_others!r} "
                f"mask={mask_others!r}"
            )

    def test_cold_cache_reproduces_the_symbolic_capture(self):
        """Known-gap check: without the prewarm, the capture reappears.

        The in-graph ``create_block_mask`` rebuild closes over the promoted
        SymInt window scalar, so at least one flex HOP carries a non-empty
        mask_mod buffer tuple — proving the assertion above discriminates the
        warmed from the cold state.  If torch ever stops promoting the int
        (making this test fail), the prewarm may no longer be load-bearing —
        re-evaluate it before deleting either side.
        """
        from nanochat import flash_attention as fa

        if not fa._use_flex():
            pytest.skip("flex path inactive (sdpa/fa3 override or unsupported)")
        torch.manual_seed(7)
        model = _build_model()
        graphs = self._trace_forward(model, prewarm=False)
        buffers = _flex_hop_other_buffers(graphs)
        assert buffers, "expected at least one flex_attention HOP in the trace"
        assert any(mask_others != () for _, mask_others in buffers), (
            "cold-cache trace no longer captures the window scalar — torch "
            "behavior changed; re-evaluate whether the prewarm is still needed"
        )
