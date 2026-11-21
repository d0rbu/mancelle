"""Tests for static shape compilation: padded batch reconstruction and fixed-dim consistency."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import torch as t

from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionLayerConfig,
)
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import (
    PackedRowShard,
    ReconstructedBatch,
    SelectionShard,
    reconstruct_batch,
    save_packed_row_shard,
    save_selection_shard,
)
from tests.encdec_test_helpers import (
    NUM_BUCKETS,
    _make_identity_selection,
    build_threshold_pipeline,
)


def _make_pipeline(n_features: int = 16384) -> FeaturePipelineConfig:
    """Build a threshold pipeline whose boundaries cover feature IDs 0..(n_features-1)."""
    boundaries = (
        t.linspace(1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1, dtype=t.float64)
        .unsqueeze(0)
        .expand(n_features, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries)


def _make_variable_shard(
    n_rows: int = 8,
    context_len: int = 32,
    seed: int = 42,
) -> PackedRowShard:
    """Create a shard with VARIABLE segments/features per row for realistic testing.

    Row i has (2 + i % 3) segments, each segment has (3 + i % 4) features.
    This produces rows with different n_seg and F_max values, which is the
    scenario that requires padding for static shapes.
    """
    t.manual_seed(seed)
    T = context_len

    tokens = t.randint(1, 32000, (n_rows, T + 1), dtype=t.uint16)
    all_seg_starts: list[int] = []
    all_seg_ends: list[int] = []
    all_seg_doc_ids: list[int] = []
    all_feat_ids: list[int] = []
    all_feat_pcts: list[int] = []
    row_seg_offsets = [0]
    seg_feat_offsets = [0]

    for r in range(n_rows):
        n_segs = 2 + r % 3  # 2, 3, or 4 segments
        seg_size = T // n_segs
        for s in range(n_segs):
            start = s * seg_size
            end = min((s + 1) * seg_size, T) if s < n_segs - 1 else T
            all_seg_starts.append(start)
            all_seg_ends.append(end)
            all_seg_doc_ids.append(0)

            n_feats = 3 + (r + s) % 4  # 3, 4, 5, or 6 features
            for _ in range(n_feats):
                all_feat_ids.append(int(t.randint(0, 16384, (1,)).item()))
                all_feat_pcts.append(int(t.randint(1, 255, (1,)).item()))
            seg_feat_offsets.append(seg_feat_offsets[-1] + n_feats)

        row_seg_offsets.append(row_seg_offsets[-1] + n_segs)

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=t.tensor(row_seg_offsets, dtype=t.int32),
        seg_token_start=t.tensor(all_seg_starts, dtype=t.uint16),
        seg_token_end=t.tensor(all_seg_ends, dtype=t.uint16),
        seg_doc_id=t.tensor(all_seg_doc_ids, dtype=t.uint8),
        seg_feat_offsets=t.tensor(seg_feat_offsets, dtype=t.int32),
        feature_ids=t.tensor(all_feat_ids, dtype=t.uint16),
        feature_pcts=t.tensor(all_feat_pcts, dtype=t.uint8),
    )


class TestPaddedReconstruction:
    """Tests for reconstruct_batch with pad_n_seg/pad_F parameters."""

    def test_padded_shapes_match_requested_dims(self):
        """Padded batch tensors must have exactly the requested dimensions."""
        shard = _make_variable_shard()
        pad_n_seg, pad_F = 10, 12

        rb = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10, pad_n_seg=pad_n_seg, pad_F=pad_F)

        assert rb.enc_fids.shape == (4, pad_n_seg, pad_F)
        assert rb.enc_pcts.shape == (4, pad_n_seg, pad_F)
        assert rb.enc_mask.shape == (4, pad_n_seg, pad_F)
        assert rb.seg_doc_id.shape == (4, pad_n_seg)

    def test_padding_positions_are_masked_false(self):
        """Padding positions in enc_mask must be False."""
        shard = _make_variable_shard()
        pad_n_seg, pad_F = 10, 12

        rb = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10, pad_n_seg=pad_n_seg, pad_F=pad_F)
        nat = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10)
        nat_n_seg = nat.enc_fids.shape[1]
        nat_F = nat.enc_fids.shape[2]

        assert not rb.enc_mask[:, nat_n_seg:, :].any(), "Padding segments must be masked False"
        assert not rb.enc_mask[:, :, nat_F:].any(), "Padding features must be masked False"

    def test_padded_seg_doc_id_is_minus_one(self):
        """Padding segments must have seg_doc_id = -1."""
        shard = _make_variable_shard()
        rb = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10, pad_n_seg=10, pad_F=12)
        nat = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10)

        assert (rb.seg_doc_id[:, nat.enc_fids.shape[1]:] == -1).all()

    def test_padded_data_matches_unpadded(self):
        """Real data region of padded batch must exactly match unpadded batch."""
        shard = _make_variable_shard()
        nat = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10)
        pad = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10, pad_n_seg=10, pad_F=12)

        # inputs, targets, seg_map are identical (same T)
        assert t.equal(nat.inputs, pad.inputs)
        assert t.equal(nat.targets, pad.targets)
        assert t.equal(nat.seg_map, pad.seg_map)

        # Real data region of encoder tensors matches
        ns, nf = nat.enc_fids.shape[1], nat.enc_fids.shape[2]
        assert t.equal(nat.enc_fids, pad.enc_fids[:, :ns, :nf])
        assert t.equal(nat.enc_pcts, pad.enc_pcts[:, :ns, :nf])
        assert t.equal(nat.enc_mask, pad.enc_mask[:, :ns, :nf])
        assert t.equal(nat.seg_doc_id[:, :ns], pad.seg_doc_id[:, :ns])

    def test_seg_map_never_indexes_padding_segments(self):
        """seg_map values must be strictly less than the actual (unpadded) segment count."""
        shard = _make_variable_shard()
        nat = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10)
        pad = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10, pad_n_seg=10, pad_F=12)

        assert t.equal(nat.seg_map, pad.seg_map)
        assert pad.seg_map.max() < nat.enc_fids.shape[1]

    def test_no_padding_when_params_omitted(self):
        """Without pad_n_seg/pad_F, behavior is unchanged (backward compat)."""
        shard = _make_variable_shard()
        rb = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10)
        assert rb.enc_fids.shape[1] <= 4  # max 4 segs in our test shard
        assert rb.enc_fids.shape[2] <= 6  # max 6 feats in our test shard


def _make_variable_selection_shard(base_shard: PackedRowShard, seed: int = 99) -> SelectionShard:
    """Create a selection shard aligned with a base shard, with variable features per segment."""
    t.manual_seed(seed)
    total_segs = base_shard.total_segments

    sel_offsets = [0]
    sel_fids: list[int] = []
    sel_buckets: list[int] = []
    sel_pcts: list[int] = []
    neg_offsets = [0]
    neg_fids: list[int] = []

    for s in range(total_segs):
        n_sel = 2 + s % 3  # 2-4 selected
        n_neg = 1 + s % 2  # 1-2 negative
        for _ in range(n_sel):
            sel_fids.append(int(t.randint(0, 16384, (1,)).item()))
            sel_buckets.append(int(t.randint(1, 9, (1,)).item()))
            sel_pcts.append(int(t.randint(1, 255, (1,)).item()))
        for _ in range(n_neg):
            neg_fids.append(int(t.randint(0, 16384, (1,)).item()))
        sel_offsets.append(sel_offsets[-1] + n_sel)
        neg_offsets.append(neg_offsets[-1] + n_neg)

    return SelectionShard(
        sel_offsets=t.tensor(sel_offsets, dtype=t.int32),
        sel_fids=t.tensor(sel_fids, dtype=t.uint16),
        sel_buckets=t.tensor(sel_buckets, dtype=t.uint8),
        sel_pcts=t.tensor(sel_pcts, dtype=t.uint8),
        neg_offsets=t.tensor(neg_offsets, dtype=t.int32),
        neg_fids=t.tensor(neg_fids, dtype=t.uint16),
    )


class TestPaddedReconstructionWithSelection:
    """Tests for reconstruct_batch_with_selection with pad_n_seg/pad_F."""

    def test_padded_shapes(self):
        base = _make_variable_shard()
        sel = _make_variable_selection_shard(base)
        pad_n_seg, pad_F = 10, 12

        result = reconstruct_batch(
            base, range(4), sel, 10,
            pad_n_seg=pad_n_seg, pad_F=pad_F,
        )
        assert result.enc_fids.shape == (4, pad_n_seg, pad_F)

    def test_padded_data_matches_unpadded(self):
        base = _make_variable_shard()
        sel = _make_variable_selection_shard(base)
        nat = reconstruct_batch(base, range(4), sel, 10)
        padded = reconstruct_batch(
            base, range(4), sel, 10, pad_n_seg=10, pad_F=12,
        )
        ns, nf = nat.enc_fids.shape[1], nat.enc_fids.shape[2]
        assert t.equal(nat.enc_fids, padded.enc_fids[:, :ns, :nf])  # enc_fids
        assert t.equal(nat.enc_mags, padded.enc_mags[:, :ns, :nf])  # enc_mags
        assert t.equal(nat.enc_mask, padded.enc_mask[:, :ns, :nf])  # enc_mask

    def test_padding_mask_false_for_selection(self):
        base = _make_variable_shard()
        sel = _make_variable_selection_shard(base)
        result = reconstruct_batch(
            base, range(4), sel, 10, pad_n_seg=10, pad_F=12,
        )
        nat = reconstruct_batch(base, range(4), sel, 10)
        ns, nf = nat.enc_fids.shape[1], nat.enc_fids.shape[2]
        assert not result.enc_mask[:, ns:, :].any()
        assert not result.enc_mask[:, :, nf:].any()


class TestCrossAttentionStaticMaxSegSize:
    """Test raw-bound behavior in the CPU SDPA reference."""

    def _make_layer(self, dec_dim=64, enc_dim=32):
        config = CrossAttentionLayerConfig(
            dec_n_embd=dec_dim,
            enc_n_embd=enc_dim,
            n_head=4,
            n_kv_head=4,
            head_dim=dec_dim // 4,
        )
        layer = CrossAttentionLayer(config)
        layer.init_weights()
        return layer

    def test_output_matches_with_explicit_max_seg_size(self):
        """Output should be identical whether max_seg_size is computed or passed.

        ``static_max_seg_size`` is a raw semantic bound and is aligned
        internally. This test passes the already-aligned value 16 as the
        legacy-compatible reference for a data-derived maximum of 6.
        """
        B, T, C = 2, 16, 64
        n_seg, F = 3, 8
        enc_dim = 32

        torch.manual_seed(42)
        layer = self._make_layer(dec_dim=C, enc_dim=enc_dim)

        decoder_hidden = torch.randn(B, T, C)
        encoder_kv = torch.randn(B, n_seg, F, enc_dim)
        enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        seg_map = torch.cat([
            torch.zeros(B, 6, dtype=torch.long),
            torch.ones(B, 5, dtype=torch.long),
            torch.full((B, 5), 2, dtype=torch.long),
        ], dim=1)

        out_auto = layer(decoder_hidden, encoder_kv, enc_mask, seg_map)
        out_static = layer(decoder_hidden, encoder_kv, enc_mask, seg_map, static_max_seg_size=16)

        assert torch.allclose(out_auto, out_static, atol=1e-5)

    def test_larger_max_seg_size_still_correct(self):
        """A max_seg_size larger than needed should still produce correct output."""
        B, T, C = 2, 16, 64
        n_seg, F = 3, 8
        enc_dim = 32

        torch.manual_seed(42)
        layer = self._make_layer(dec_dim=C, enc_dim=enc_dim)

        decoder_hidden = torch.randn(B, T, C)
        encoder_kv = torch.randn(B, n_seg, F, enc_dim)
        enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        seg_map = torch.cat([
            torch.zeros(B, 6, dtype=torch.long),
            torch.ones(B, 5, dtype=torch.long),
            torch.full((B, 5), 2, dtype=torch.long),
        ], dim=1)

        out_auto = layer(decoder_hidden, encoder_kv, enc_mask, seg_map)
        out_large = layer(decoder_hidden, encoder_kv, enc_mask, seg_map, static_max_seg_size=64)

        assert torch.allclose(out_auto, out_large, atol=1e-5)


def _write_shards_with_selection(
    output_dir: Path,
    n_shards: int = 2,
    max_n_seg: int = 10,
    max_seg_size: int = 32,
    **shard_kwargs,
) -> tuple[list[Path], Path]:
    """Write base shards + selection shards + manifest for static shape testing.

    Returns (base_shard_paths, selection_dir).
    Static shapes require selection shards — base shards alone have ~8k
    raw features which are not padded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sel_dir = output_dir / "sel"
    sel_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_shards):
        base = _make_variable_shard(seed=42 + i, **shard_kwargs)
        path = save_packed_row_shard(base, output_dir / f"shard_{i:05d}")
        paths.append(path)
        # Create matching selection shard
        sel = _make_variable_selection_shard(base, seed=99 + i)
        save_selection_shard(sel, sel_dir / f"shard_{i:05d}")

    manifest = {
        "checksum": "test",
        "kind": "percentile",
        "metadata": {
            "type": "packed_rows",
            "max_n_seg": max_n_seg,
            "max_seg_size": max_seg_size,
            "context_len": shard_kwargs.get("context_len", 32),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest))
    write_manifest(
        sel_dir,
        "test-selection",
        OverlayKind.PACKED_ROW,
        metadata={
            "packed_rows": {
                "packed_row_fingerprint": packed_row_fingerprint(output_dir),
            },
        },
    )
    return paths, sel_dir


class TestPrecomputedLoaderStaticShapes:
    """Tests that precomputed_data_loader uses selection shards for static padding."""

    def test_all_batches_same_shape_with_selection(self, tmp_path):
        """With selection shards, every batch must have identical shapes."""
        _, sel_dir = _write_shards_with_selection(
            tmp_path / "packed", n_shards=2, n_rows=8,
            max_n_seg=10, max_seg_size=32,
        )
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            selection_dir=sel_dir,
        )

        shapes = set()
        for i, batch in enumerate(loader):
            if i >= 8:
                break
            enc_fids = batch[2]
            shapes.add(enc_fids.shape)

        assert len(shapes) == 1, f"Expected 1 unique shape, got {len(shapes)}: {shapes}"

    def test_pad_n_seg_matches_manifest(self, tmp_path):
        """n_seg dim must match manifest max_n_seg."""
        _, sel_dir = _write_shards_with_selection(
            tmp_path / "packed", n_shards=1, n_rows=8,
            max_n_seg=10,
        )
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            selection_dir=sel_dir,
        )
        batch = next(loader)
        enc_fids = batch[2]
        assert enc_fids.shape[1] == 10  # max_n_seg from manifest

    def test_pad_F_from_selection_shards(self, tmp_path):
        """F dim must come from selection shard scan, not base manifest."""
        _, sel_dir = _write_shards_with_selection(
            tmp_path / "packed", n_shards=1, n_rows=8,
            max_n_seg=10,
        )
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            selection_dir=sel_dir,
        )
        batch = next(loader)
        enc_fids = batch[2]
        # Selection shards have 2-4 selected + 1-2 negative = 3-6 total per seg
        # pad_F should be the global max across all segments
        assert enc_fids.shape[2] >= 3  # min possible
        assert enc_fids.shape[2] <= 6  # max possible from _make_variable_selection_shard

    def test_no_selection_shards_means_no_static_shapes(self, tmp_path):
        """Without selection shards, shapes should be per-batch (variable)."""
        output_dir = tmp_path / "packed"
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            shard = _make_variable_shard(seed=42 + i)
            save_packed_row_shard(shard, output_dir / f"shard_{i:05d}")
        # Write manifest with max_n_seg but no selection shards
        manifest = {
            "checksum": "test",
            "kind": "percentile",
            "metadata": {
                "type": "packed_rows",
                "max_n_seg": 10,
                "max_seg_size": 32,
            },
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest))

        loader = precomputed_data_loader(
            packed_row_dir=output_dir,
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )
        batch = next(loader)
        # Without selection shards, n_seg and F should NOT be padded to manifest values
        enc_fids = batch[2]
        assert enc_fids.shape[1] <= 4, "Should not pad n_seg without selection shards"

    def test_max_seg_size_in_state_dict(self, tmp_path):
        """State dict should include max_seg_size when selection shards available."""
        _, sel_dir = _write_shards_with_selection(
            tmp_path / "packed", n_shards=1, n_rows=8,
            max_n_seg=10, max_seg_size=32,
        )
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            selection_dir=sel_dir,
        )
        batch = next(loader)
        state = batch[9]  # index 9: state_dict (index 8 is fp_targets)
        assert state.get("max_seg_size") == 32


class TestCompileStaticResolution:
    """Tests for compile_static config -> dynamic flag resolution."""

    def test_auto_with_static_shapes_returns_false(self):
        """auto + static shapes available -> dynamic=False."""
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic("auto", mode="pretrain", static_shapes_available=True) is False

    def test_auto_without_static_shapes_returns_true(self):
        """auto + no static shapes -> dynamic=True."""
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic("auto", mode="pretrain", static_shapes_available=False) is True

    def test_true_forces_false(self):
        """compile_static=true -> dynamic=False regardless."""
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic(True, mode="pretrain", static_shapes_available=False) is False

    def test_false_forces_true(self):
        """compile_static=false -> dynamic=True regardless."""
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic(False, mode="pretrain", static_shapes_available=True) is True

    def test_compile_call_has_no_mode(self):
        """The torch.compile call in build_shared_state never sets mode=.

        CUDA graphs (mode='reduce-overhead') + FP8 causes a SIGSEGV on first
        graph capture. The compile block must always use the default mode, which
        matches the base decoder's torch.compile(model, dynamic=False) call.

        This test inspects the source of build_shared_state to assert that the
        compile call contains no mode= keyword argument — catching any future
        reintroduction of reduce-overhead or other explicit modes.
        """
        from scripts.train.conditional_encdec_train import build_shared_state

        src = inspect.getsource(build_shared_state)
        # The compile call must be exactly this form — no mode= kwarg
        assert "t.compile(model, dynamic=dynamic)" in src, (
            "Expected 't.compile(model, dynamic=dynamic)' with no mode= argument. "
            "Adding mode='reduce-overhead' causes SIGSEGV with FP8 on first graph capture."
        )
        # Belt-and-suspenders: confirm no line containing .compile( also sets mode=
        import re
        for line in src.splitlines():
            if re.search(r"\bt\.compile\(|torch\.compile\(", line):
                assert "mode=" not in line, (
                    f"Found mode= in compile call — will cause SIGSEGV with FP8: {line.strip()!r}"
                )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile+FP8")
    def test_fp8_plus_default_compile_no_crash(self):
        """FP8 conversion + torch.compile(dynamic=False) must not crash on first forward.

        This is the exact combination that caused the SIGSEGV in PR #294: the
        live dataloader threaded pad_F through, which triggered reduce-overhead
        (CUDA graphs), which segfaulted during first graph capture with FP8.
        Default compile mode (no CUDA graphs) must be safe with FP8.
        """

        from nanochat.fp8_utils import apply_fp8_training
        from nanochat.gpt import GPTConfig
        from steering.conditional.encoder_decoder.config import (
            CrossAttentionConfig,
            EncoderConfig,
        )
        from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

        device = "cuda"
        B, T_len, N_SEG, F_MAX = 2, 128, 2, 8
        VOCAB = 256
        dec_cfg = GPTConfig(
            sequence_len=T_len, vocab_size=VOCAB,
            n_layer=2, n_head=4, n_kv_head=4, n_embd=256, window_pattern="L",
        )
        enc_cfg = EncoderConfig(n_embd=128, depth=1, head_dim=64)
        xattn_cfg = CrossAttentionConfig(frequency=2)
        with torch.device("meta"):
            model = EncoderDecoderGPT(dec_cfg, enc_cfg, xattn_cfg, n_sae_features=16384, num_buckets=10)
        model.to_empty(device=device)
        model.init_weights()

        apply_fp8_training(model, recipe="tensorwise")
        # Must match the exact compile call in build_shared_state — no mode=
        compiled = torch.compile(model, dynamic=False)

        torch.manual_seed(0)
        x = torch.randint(0, VOCAB, (B, T_len), device=device)
        y = torch.randint(0, VOCAB, (B, T_len), device=device)
        enc_fids = torch.randint(0, 16384, (B, N_SEG, F_MAX), device=device)
        enc_mags = torch.zeros(B, N_SEG, F_MAX, dtype=torch.int64, device=device)
        enc_mask = torch.ones(B, N_SEG, F_MAX, dtype=torch.bool, device=device)
        seg_map = (torch.arange(T_len, device=device).unsqueeze(0).expand(B, -1) // (T_len // N_SEG))

        # Two forward passes: first triggers graph warmup/capture, second replays
        for _ in range(2):
            loss = compiled(x, y, enc_fids, enc_mags, enc_mask, seg_map)
            if not isinstance(loss, torch.Tensor):
                loss = loss.loss
            assert torch.isfinite(loss), f"Non-finite loss from FP8+compiled model: {loss}"


class TestPrecomputeTracksMaxDims:
    """Tests that the precompute script's _compute_shard_max_dims works correctly."""

    def test_compute_shard_max_dims(self):
        """_compute_shard_max_dims returns (max_n_seg, max_seg_size) — no max_F."""
        from scripts.preprocess.precompute_packed_rows import _compute_shard_max_dims

        shard = _make_variable_shard(n_rows=8)
        result = _compute_shard_max_dims(shard)

        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"
        max_n_seg, max_seg_size = result

        assert max_n_seg >= 2  # rows have 2-4 segs
        assert max_n_seg <= 4
        assert max_seg_size >= 1
        assert max_seg_size <= 32  # T=32, min 2 segs -> max seg = 16


class TestBackfillScript:
    """Tests for the backfill script that adds shape metadata to existing manifests."""

    def test_backfill_adds_shape_metadata(self, tmp_path):
        from scripts.preprocess.backfill_packed_row_shapes import backfill_packed_row_shapes

        output_dir = tmp_path / "packed"
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            shard = _make_variable_shard(seed=42 + i, n_rows=8)
            save_packed_row_shard(shard, output_dir / f"shard_{i:05d}")

        # Write manifest WITHOUT shape metadata
        manifest = {
            "checksum": "test",
            "kind": "pct",
            "metadata": {
                "type": "packed_rows",
                "context_len": 32,
                "buffer_size": 1,
                "rows_per_shard": 8,
                "pct_checksum": "pct-test",
                "tok_checksum": "tok-test",
            },
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest))

        # Run backfill
        backfill_packed_row_shapes(output_dir)

        # Verify metadata now has shapes
        updated = json.loads((output_dir / "manifest.json").read_text())
        meta = updated["metadata"]
        assert "max_n_seg" in meta
        assert "max_seg_size" in meta
        # max_F should NOT be in the manifest — it comes from selection shards
        assert "max_F" not in meta
        assert meta["max_n_seg"] >= 2
        assert meta["max_seg_size"] >= 1

def _make_small_model(n_sae=256, num_buckets=10, with_feature_pred=True):
    """Create a small EncoderDecoderGPT for testing."""
    dec = GPTConfig(
        sequence_len=32, vocab_size=512, n_layer=4, n_head=4, n_kv_head=4,
        n_embd=64, window_pattern="SL",
    )
    enc = EncoderConfig(n_embd=32, depth=2, n_head=2, n_kv_head=2, head_dim=16)
    xattn = CrossAttentionConfig(frequency=2)
    fp = (
        FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.BEST_TOKEN,
            loss_fn=FeaturePredLossFn.BCE,
        )
        if with_feature_pred
        else None
    )
    model = EncoderDecoderGPT(dec, enc, xattn, n_sae, num_buckets, feature_pred_config=fp)
    model.init_weights()
    return model


class TestPaddedModelEquivalence:
    """Verify that padding doesn't change model outputs."""

    def test_loss_matches_padded_vs_unpadded(self):
        """Forward pass loss must match between padded and unpadded inputs."""
        torch.manual_seed(42)
        model = _make_small_model(with_feature_pred=False)
        model.eval()

        B, T = 2, 32
        n_seg_real, F_real = 3, 5
        n_seg_pad, F_pad = 6, 10

        idx = torch.randint(0, 512, (B, T))
        targets = torch.randint(0, 512, (B, T))
        enc_fids = torch.randint(0, 256, (B, n_seg_real, F_real))
        enc_mags = torch.randint(0, 10, (B, n_seg_real, F_real))
        enc_mask = torch.ones(B, n_seg_real, F_real, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 11:22] = 1
        seg_map[:, 22:] = 2
        seg_doc_id = torch.zeros(B, n_seg_real, dtype=torch.long)

        with torch.no_grad():
            loss_nat = model(
                idx, targets, enc_fids, enc_mags, enc_mask, seg_map,
                seg_doc_id=seg_doc_id,
            )

        # Padded
        enc_fids_pad = torch.zeros(B, n_seg_pad, F_pad, dtype=torch.long)
        enc_mags_pad = torch.zeros(B, n_seg_pad, F_pad, dtype=torch.long)
        enc_mask_pad = torch.zeros(B, n_seg_pad, F_pad, dtype=torch.bool)
        seg_doc_id_pad = torch.full((B, n_seg_pad), -1, dtype=torch.long)

        enc_fids_pad[:, :n_seg_real, :F_real] = enc_fids
        enc_mags_pad[:, :n_seg_real, :F_real] = enc_mags
        enc_mask_pad[:, :n_seg_real, :F_real] = enc_mask
        seg_doc_id_pad[:, :n_seg_real] = seg_doc_id

        with torch.no_grad():
            loss_pad = model(
                idx, targets, enc_fids_pad, enc_mags_pad, enc_mask_pad, seg_map,
                seg_doc_id=seg_doc_id_pad,
            )

        assert torch.allclose(loss_nat, loss_pad, atol=1e-5), (
            f"Loss mismatch: natural={loss_nat.item():.6f}, padded={loss_pad.item():.6f}"
        )

    def test_gradients_match_padded_vs_unpadded(self):
        """Parameter gradients must match between padded and unpadded inputs."""
        import nanochat.flash_attention as _fa_mod
        from nanochat.flash_attention import FlashAttnImpl

        # Force SDPA backend: flex_attention's compiled backward fails on CPU
        # in inductor with dynamic=False (tuple index out of range in codegen).
        _orig_override = _fa_mod._override_impl
        _fa_mod._override_impl = FlashAttnImpl.SDPA
        try:
            torch.manual_seed(42)
            B, T = 2, 32
            n_seg_real, F_real = 3, 5
            n_seg_pad, F_pad = 6, 10

            idx = torch.randint(0, 512, (B, T))
            targets = torch.randint(0, 512, (B, T))
            enc_fids = torch.randint(0, 256, (B, n_seg_real, F_real))
            enc_mags = torch.randint(0, 10, (B, n_seg_real, F_real))
            enc_mask = torch.ones(B, n_seg_real, F_real, dtype=torch.bool)
            seg_map = torch.zeros(B, T, dtype=torch.long)
            seg_map[:, 11:22] = 1
            seg_map[:, 22:] = 2
            seg_doc_id = torch.zeros(B, n_seg_real, dtype=torch.long)

            # Unpadded
            model1 = _make_small_model(with_feature_pred=False)
            model1.init_weights()
            loss1 = model1(
                idx, targets, enc_fids, enc_mags, enc_mask, seg_map,
                seg_doc_id=seg_doc_id,
            )
            loss1.backward()
            grads1 = {n: p.grad.clone() for n, p in model1.named_parameters() if p.grad is not None}

            # Padded — same architecture and weights as model1
            model2 = _make_small_model(with_feature_pred=False)
            model2.load_state_dict(model1.state_dict())
            enc_fids_pad = torch.zeros(B, n_seg_pad, F_pad, dtype=torch.long)
            enc_mags_pad = torch.zeros(B, n_seg_pad, F_pad, dtype=torch.long)
            enc_mask_pad = torch.zeros(B, n_seg_pad, F_pad, dtype=torch.bool)
            seg_doc_id_pad = torch.full((B, n_seg_pad), -1, dtype=torch.long)
            enc_fids_pad[:, :n_seg_real, :F_real] = enc_fids
            enc_mags_pad[:, :n_seg_real, :F_real] = enc_mags
            enc_mask_pad[:, :n_seg_real, :F_real] = enc_mask
            seg_doc_id_pad[:, :n_seg_real] = seg_doc_id
            loss2 = model2(
                idx, targets, enc_fids_pad, enc_mags_pad, enc_mask_pad, seg_map,
                seg_doc_id=seg_doc_id_pad,
            )
            loss2.backward()
            grads2 = {n: p.grad.clone() for n, p in model2.named_parameters() if p.grad is not None}

            for name, grad1 in grads1.items():
                assert torch.allclose(grad1, grads2[name], atol=1e-5), (
                    f"Gradient mismatch for {name}: max diff={(grad1 - grads2[name]).abs().max():.2e}"
                )
        finally:
            _fa_mod._override_impl = _orig_override

    def test_feature_prediction_excludes_padding(self):
        """Feature prediction loss should not include padding segments."""
        torch.manual_seed(42)
        model = _make_small_model(with_feature_pred=True)
        n_sae = model.feature_pred_head.out_features
        model.eval()

        B, T = 2, 32
        n_seg_real, F_real = 3, 5
        n_seg_pad, F_pad = 6, 10

        idx = torch.randint(0, 512, (B, T))
        targets = torch.randint(0, 512, (B, T))
        enc_fids = torch.randint(0, 256, (B, n_seg_real, F_real))
        enc_mags = torch.randint(0, 10, (B, n_seg_real, F_real))
        enc_mask = torch.ones(B, n_seg_real, F_real, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 11:22] = 1
        seg_map[:, 22:] = 2
        seg_doc_id = torch.zeros(B, n_seg_real, dtype=torch.long)
        fp_targets = torch.rand(B, n_seg_real, n_sae)

        with torch.no_grad():
            result_nat = model(
                idx, targets, enc_fids, enc_mags, enc_mask, seg_map,
                seg_doc_id=seg_doc_id, fp_targets=fp_targets,
            )

        enc_fids_pad = torch.zeros(B, n_seg_pad, F_pad, dtype=torch.long)
        enc_mags_pad = torch.zeros(B, n_seg_pad, F_pad, dtype=torch.long)
        enc_mask_pad = torch.zeros(B, n_seg_pad, F_pad, dtype=torch.bool)
        seg_doc_id_pad = torch.full((B, n_seg_pad), -1, dtype=torch.long)
        fp_targets_pad = torch.zeros(B, n_seg_pad, n_sae)
        enc_fids_pad[:, :n_seg_real, :F_real] = enc_fids
        enc_mags_pad[:, :n_seg_real, :F_real] = enc_mags
        enc_mask_pad[:, :n_seg_real, :F_real] = enc_mask
        seg_doc_id_pad[:, :n_seg_real] = seg_doc_id
        fp_targets_pad[:, :n_seg_real] = fp_targets

        with torch.no_grad():
            result_pad = model(
                idx, targets, enc_fids_pad, enc_mags_pad, enc_mask_pad, seg_map,
                seg_doc_id=seg_doc_id_pad, fp_targets=fp_targets_pad,
            )

        # ForwardResult: (loss, lm_loss, feature_pred_loss)
        assert torch.allclose(result_nat.lm_loss, result_pad.lm_loss, atol=1e-5)
        assert torch.allclose(result_nat.feature_pred_loss, result_pad.feature_pred_loss, atol=1e-5)

    def test_max_seg_size_produces_same_output(self):
        """Passing static_max_seg_size to forward should not change output."""
        torch.manual_seed(42)
        model = _make_small_model(with_feature_pred=False)
        model.eval()

        B, T = 2, 32
        n_seg, F = 3, 5
        idx = torch.randint(0, 512, (B, T))
        targets = torch.randint(0, 512, (B, T))
        enc_fids = torch.randint(0, 256, (B, n_seg, F))
        enc_mags = torch.randint(0, 10, (B, n_seg, F))
        enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 11:22] = 1
        seg_map[:, 22:] = 2

        with torch.no_grad():
            loss_auto = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
            loss_static = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map, static_max_seg_size=32)

        assert torch.allclose(loss_auto, loss_static, atol=1e-5)


class TestStaticShapeIntegration:
    """Integration tests: full loader -> model with static shapes (requires selection shards)."""

    def test_100_batches_identical_shapes(self, tmp_path):
        """All batches from padded loader must have identical tensor shapes."""
        _, sel_dir = _write_shards_with_selection(
            tmp_path / "packed", n_shards=3, n_rows=16,
            max_n_seg=10, max_seg_size=32,
        )
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed", B=4, device="cpu",
            feature_pipeline=_make_pipeline(),
            selection_dir=sel_dir,
        )

        first_shapes = None
        for i, batch in enumerate(loader):
            if i >= 100:
                break
            shapes = tuple(
                tensor.shape for tensor in batch[:8]
                if isinstance(tensor, t.Tensor)
            )
            if first_shapes is None:
                first_shapes = shapes
            else:
                assert shapes == first_shapes, (
                    f"Batch {i} shapes {shapes} != first batch shapes {first_shapes}"
                )

    def test_cross_shard_boundary_preserves_static_shapes(self, tmp_path):
        """Batches spanning shard boundaries must still have static shapes."""
        _, sel_dir = _write_shards_with_selection(
            tmp_path / "packed", n_shards=2, n_rows=5,
            max_n_seg=10, max_seg_size=32,
        )
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed", B=4, device="cpu",
            feature_pipeline=_make_pipeline(),
            selection_dir=sel_dir,
        )

        shapes = set()
        for i, batch in enumerate(loader):
            if i >= 5:
                break
            shapes.add(batch[2].shape)

        assert len(shapes) == 1, f"Expected 1 unique shape across shard boundary, got {shapes}"

    def test_model_forward_no_crash(self, tmp_path):
        """10 forward passes with padded static shapes, no NaN."""
        _, sel_dir = _write_shards_with_selection(
            tmp_path / "packed", n_shards=1, n_rows=16,
            max_n_seg=10, max_seg_size=32,
            context_len=32,
        )
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed", B=4, device="cpu",
            feature_pipeline=_make_pipeline(),
            selection_dir=sel_dir,
        )
        # Use vocab_size=32000 and n_sae=16384 to match shard data ranges
        dec = GPTConfig(
            sequence_len=32, vocab_size=32000, n_layer=4, n_head=4, n_kv_head=4,
            n_embd=64, window_pattern="SL",
        )
        enc = EncoderConfig(n_embd=32, depth=2, n_head=2, n_kv_head=2, head_dim=16)
        xattn = CrossAttentionConfig(frequency=2)
        # num_buckets=11: selection shards use effective_buckets=10 for negatives,
        # so the magnitude embedding needs at least 11 entries (indices 0-10)
        model = EncoderDecoderGPT(dec, enc, xattn, n_sae_features=16384, num_buckets=11, feature_pred_config=None)
        model.init_weights()
        model.eval()

        for i, batch in enumerate(loader):
            if i >= 10:
                break
            inputs, targets, enc_fids, enc_mags, enc_mask, _enc_pcts, seg_map, seg_doc_id, _fp_targets, state = batch
            max_seg_size = state.get("max_seg_size")
            with torch.no_grad():
                loss = model(inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, static_max_seg_size=max_seg_size)
            assert not torch.isnan(loss), f"NaN loss at batch {i}"
            assert not torch.isinf(loss), f"Inf loss at batch {i}"


class TestStaticShapeDDP:
    """Tests for static shape behavior under simulated DDP conditions."""

    @pytest.mark.slow
    def test_all_ranks_see_same_shapes(self, tmp_path):
        """Simulate multi-rank: each rank gets same static shapes from selection shards."""
        _, sel_dir = _write_shards_with_selection(
            tmp_path / "packed", n_shards=4, n_rows=8,
            max_n_seg=10, max_seg_size=32,
        )

        loader_r0 = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed", B=4, device="cpu",
            feature_pipeline=_make_pipeline(),
            selection_dir=sel_dir,
        )
        loader_r1 = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed", B=4, device="cpu",
            feature_pipeline=_make_pipeline(),
            selection_dir=sel_dir,
        )

        for i, (b0, b1) in enumerate(zip(loader_r0, loader_r1, strict=True)):
            if i >= 10:
                break
            assert b0[2].shape == b1[2].shape, (
                f"Rank shape mismatch at batch {i}: {b0[2].shape} vs {b1[2].shape}"
            )


# ---------------------------------------------------------------------------
# Coverage gap tests
# ---------------------------------------------------------------------------


class TestPaddingAssertionErrors:
    """Verify assertion errors when pad values are too small."""

    def test_pad_n_seg_too_small_raises(self):
        shard = _make_variable_shard()
        with pytest.raises(AssertionError, match="pad_n_seg=1 < batch_n_seg"):
            reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10, pad_n_seg=1)

    def test_pad_F_too_small_raises(self):
        shard = _make_variable_shard()
        with pytest.raises(AssertionError, match="pad_F=1 < batch_f_max"):
            reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10, pad_F=1)

    def test_selection_pad_n_seg_too_small_raises(self):
        base = _make_variable_shard()
        sel = _make_variable_selection_shard(base)
        with pytest.raises(AssertionError, match="pad_n_seg=1 < batch_n_seg"):
            reconstruct_batch(base, range(4), sel, 10, pad_n_seg=1)

    def test_selection_pad_F_too_small_raises(self):
        base = _make_variable_shard()
        sel = _make_variable_selection_shard(base)
        with pytest.raises(AssertionError, match="pad_F=1 < batch_f_max"):
            reconstruct_batch(base, range(4), sel, 10, pad_F=1)


class TestPaddingEdgeCases:
    """Edge cases: exact-match padding, single-row batches."""

    def test_pad_equal_to_natural_dims(self):
        """When pad values exactly match natural dims, output equals unpadded."""
        shard = _make_variable_shard()
        nat = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10)
        nat_ns, nat_f = nat.enc_fids.shape[1], nat.enc_fids.shape[2]

        padded = reconstruct_batch(shard, range(4), _make_identity_selection(shard), 10, pad_n_seg=nat_ns, pad_F=nat_f)
        # Shapes should be identical across all fields
        for field in ReconstructedBatch._fields:
            nat_val = getattr(nat, field)
            pad_val = getattr(padded, field)
            assert nat_val.shape == pad_val.shape, f"Shape mismatch for field {field}"
            assert t.equal(nat_val, pad_val), f"Data mismatch for field {field}"

    def test_single_row_batch_padded(self):
        """Single-row batches should work with padding."""
        shard = _make_variable_shard()
        result = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10, pad_n_seg=10, pad_F=12)
        assert result.enc_fids.shape == (1, 10, 12)

    def test_single_row_selection_padded(self):
        """Single-row selection batches should work with padding."""
        base = _make_variable_shard()
        sel = _make_variable_selection_shard(base)
        result = reconstruct_batch(
            base, [0], sel, 10, pad_n_seg=10, pad_F=12,
        )
        assert result.enc_fids.shape == (1, 10, 12)


class TestLoaderEdgeCases:
    """Loader edge cases: missing manifest, empty selection dir."""

    def test_missing_manifest_still_works(self, tmp_path):
        """Loader works without manifest.json (variable shapes, no padding)."""
        output_dir = tmp_path / "packed"
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            shard = _make_variable_shard(seed=42 + i)
            save_packed_row_shard(shard, output_dir / f"shard_{i:05d}")
        # No manifest.json at all

        loader = precomputed_data_loader(packed_row_dir=output_dir, B=4, feature_pipeline=_make_pipeline(), device="cpu")
        batch = next(loader)
        assert batch is not None
        # state_dict should have max_seg_size=None
        assert batch[9].get("max_seg_size") is None  # index 9: state_dict (index 8 is fp_targets)

    def test_empty_selection_dir_raises(self, tmp_path):
        """Explicit empty selection_dir is invalid under manifest validation."""
        output_dir = tmp_path / "packed"
        sel_dir = output_dir / "sel"
        output_dir.mkdir(parents=True, exist_ok=True)
        sel_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            shard = _make_variable_shard(seed=42 + i)
            save_packed_row_shard(shard, output_dir / f"shard_{i:05d}")
        # Write manifest with max_n_seg
        manifest = {"checksum": "test", "kind": "percentile",
                    "metadata": {"max_n_seg": 10, "max_seg_size": 32}}
        (output_dir / "manifest.json").write_text(json.dumps(manifest))

        loader = precomputed_data_loader(
            packed_row_dir=output_dir, B=4, feature_pipeline=_make_pipeline(), device="cpu", selection_dir=sel_dir,
        )
        with pytest.raises(ValueError, match="contains no selection shards"):
            next(loader)


class TestCompileStaticStringVariants:
    """Test string variants of compile_static from YAML."""

    def test_string_true(self):
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic("true", mode="pretrain", static_shapes_available=False) is False

    def test_string_false(self):
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic("false", mode="pretrain", static_shapes_available=True) is True


class TestBackfillEdgeCases:
    """Backfill script edge cases."""

    def test_no_shards_raises(self, tmp_path):
        from scripts.preprocess.backfill_packed_row_shapes import backfill_packed_row_shapes

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No packed row shards"):
            backfill_packed_row_shapes(empty_dir)


class TestSelectFeaturesForShard:
    """Tests for the selection shard precomputation core function."""

    def test_select_features_produces_valid_shard(self, tmp_path):
        """select_features_for_shard produces a valid SelectionShard."""
        from scripts.preprocess.precompute_selected_features import select_features_for_shard
        from steering.conditional.feature_pipeline import FeaturePipelineConfig
        from steering.conditional.sampling import SamplingBundle, SamplingSelection
        from steering.overlay_io import overlay_shard_path
        from steering.packed_batch_io import load_selection_shard

        n_sae = 256
        num_buckets = 10

        # Build a sampling pipeline with synthetic percentile table
        boundaries = t.linspace(0.0, 1.0, num_buckets - 1).unsqueeze(0).expand(n_sae, -1).to(t.float64)
        percentile_table = t.rand(n_sae, 99, dtype=t.float64)
        rng = t.Generator().manual_seed(42)
        pipeline = FeaturePipelineConfig(
            selection_config=SamplingBundle(
                selection=SamplingSelection(mean=3.0, std=3.0, min_k=1, max_k=10),
                percentile_table=percentile_table,
                rng=rng,
            ),
            boundaries=boundaries,
            negative_bundle=None,
            effective_buckets=num_buckets,
        )

        # Create a base shard with feature IDs clamped to [0, n_sae)
        shard = _make_variable_shard(n_rows=4, context_len=32, seed=42)
        shard = PackedRowShard(
            tokens=shard.tokens,
            row_seg_offsets=shard.row_seg_offsets,
            seg_token_start=shard.seg_token_start,
            seg_token_end=shard.seg_token_end,
            seg_doc_id=shard.seg_doc_id,
            seg_feat_offsets=shard.seg_feat_offsets,
            feature_ids=(shard.feature_ids.to(t.int32) % n_sae).to(t.uint16),
            feature_pcts=shard.feature_pcts,
        )
        base_path = save_packed_row_shard(shard, tmp_path / "shard_00000")

        # Run selection
        output_dir = tmp_path / "sel"
        output_dir.mkdir()
        result = select_features_for_shard(base_path, pipeline, output_dir, "shard_00000")
        assert result is True

        # Load and verify
        sel = load_selection_shard(overlay_shard_path(output_dir, "shard_00000"))
        assert sel.total_segments == shard.total_segments
        assert sel.total_selected > 0
        if len(sel.sel_fids) > 0:
            assert sel.sel_fids.to(t.int32).max().item() < n_sae
        if len(sel.sel_buckets) > 0:
            assert sel.sel_buckets.to(t.int32).max().item() < num_buckets
