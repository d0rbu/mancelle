"""Compact and clustered feature-prediction targets for encoder-decoder SFT."""

from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch as t
from omegaconf import OmegaConf

import nanochat.flash_attention as _fa_mod
from errors import FeatureClustersError, FeatureVocabError
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from scripts.preprocess.convert_sft_to_parquet import (
    prepare_messages_for_full_nanochat_render,
    render_messages_to_full_nanochat_text,
)
from scripts.train import conditional_encdec_sft as sft
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT, ForwardResult
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    assert_resume_feature_clusters_match,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, canonical_dict
from steering.conditional.feature_vocab import (
    FEATURE_EMBED_KEY,
    FeatureVocab,
    assert_resume_feature_vocab_matches,
)
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.refined_types import PositiveInt
from tests.steering.goldens.gen_encdec_sft_compact_fp_targets_legacy_golden import (
    GOLDEN_PATH,
    ORACLE_COMMIT,
    capture_legacy_fp_targets,
)
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    _full_sequence_char_count,
    patch_load_sft_features,
)
from tests.steering.test_encdec_sft_feature_prediction import (
    _make_synthetic_percentile_table,
)


@pytest.fixture(autouse=True)
def _force_sdpa_backend():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_legacy_fp_target_golden_was_captured_at_the_1279_tip() -> None:
    assert _golden()["oracle_commit"] == ORACLE_COMMIT


def test_full_width_fp_targets_are_bitwise_equal_to_the_base_capture() -> None:
    golden = _golden()["fp_targets"]
    actual = capture_legacy_fp_targets()
    expected = t.zeros(golden["shape"], dtype=t.float32)
    for index, value in zip(
        golden["nonzero_indices"],
        golden["nonzero_values"],
        strict=True,
    ):
        expected[tuple(index)] = value

    assert t.equal(actual, expected)
    assert hashlib.sha256(actual.view(t.uint8).numpy().tobytes()).hexdigest() == golden["sha256"]


N_SAE = 8
NUM_BUCKETS = 4


def _resolved_clusters(
    tmp_path: Path,
    groups: tuple[tuple[int, ...], ...],
) -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(clusters=groups)
    descriptor = FeatureClusterDescriptor.from_definition(
        definition,
        n_sae_features=N_SAE,
    )
    return ResolvedFeatureClusters(
        source_path=tmp_path / "clusters.yaml",
        definition=definition,
        descriptor=descriptor,
    )


def _target_layout(
    *,
    feature_mask: FeatureMask | None = None,
    resolved_clusters: ResolvedFeatureClusters | None = None,
) -> tuple[FeaturePredictionTargetLayout, FeatureClusterConsolidation | None]:
    consolidation = (
        FeatureClusterConsolidation.build(resolved_clusters, feature_mask)
        if resolved_clusters is not None
        else None
    )
    vocab = effective_feature_vocab(
        consolidation if consolidation is not None else feature_mask,
        N_SAE,
    )
    layout = FeaturePredictionTargetLayout.build(
        n_sae_features=PositiveInt.parse(N_SAE),
        feature_mask=feature_mask,
        feature_clusters=consolidation,
        feature_vocab=vocab,
    )
    return layout, consolidation


def _custom_features(
    task: SingleTurnTask,
    tokenizer: SFTMockTokenizer,
    segment_ids: tuple[tuple[int, ...], ...],
    segment_magnitudes: tuple[tuple[float, ...], ...],
) -> dict[int, list[SegmentFeatures]]:
    assert len(segment_ids) == len(segment_magnitudes)
    features: dict[int, list[SegmentFeatures]] = {}
    for example_idx in range(len(task)):
        total_chars = _full_sequence_char_count(tokenizer, task[example_idx])
        segment_size = total_chars // len(segment_ids)
        segments: list[SegmentFeatures] = []
        for segment_idx, (ids, magnitudes) in enumerate(
            zip(segment_ids, segment_magnitudes, strict=True)
        ):
            segments.append(
                SegmentFeatures(
                    segment_idx=segment_idx,
                    position=segment_idx,
                    feature_ids=t.tensor(ids, dtype=t.int32),
                    feature_mags=t.tensor(magnitudes, dtype=t.float32),
                    char_start=segment_idx * segment_size,
                    char_end=(
                        total_chars
                        if segment_idx == len(segment_ids) - 1
                        else (segment_idx + 1) * segment_size
                    ),
                )
            )
        features[example_idx] = segments
    return features


def _probe_texts(task: SingleTurnTask, tokenizer: SFTMockTokenizer) -> dict[int, str]:
    """Render every task conversation with the exact converter helper.

    ``make_sft_data_generator`` runs a startup content-alignment probe (#1298)
    that byte-compares ``store.load_texts(indices)`` against a live re-render
    of the task via ``render_messages_to_full_nanochat_text`` — the same
    helper ``convert_mixture_to_parquet`` uses to write parquet ``text``
    columns. Reusing it here (rather than hand-rolling a matching string)
    means a real divergence between the synthetic store and the task would
    be caught, not trivially satisfied.
    """
    texts: dict[int, str] = {}
    for example_idx in range(len(task)):
        messages, skip_reason = prepare_messages_for_full_nanochat_render(task[example_idx])
        assert messages is not None, (
            f"synthetic conversation {example_idx} unexpectedly skipped: {skip_reason}"
        )
        texts[example_idx] = render_messages_to_full_nanochat_text(tokenizer, messages)
    return texts


def _compact_batch(
    *,
    layout: FeaturePredictionTargetLayout,
    consolidation: FeatureClusterConsolidation | None,
    feature_mask: FeatureMask | None,
    segment_ids: tuple[tuple[int, ...], ...],
    segment_magnitudes: tuple[tuple[float, ...], ...],
) -> tuple:
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=4, user_chars=3, asst_chars=8, prefix="c")
    features = _custom_features(task, tokenizer, segment_ids, segment_magnitudes)
    sequence_len = len(tokenizer.render_conversation(task[0])[0]) - 1
    boundaries = (
        t.tensor([25.0, 50.0, 75.0], dtype=t.float32).unsqueeze(0).expand(N_SAE, -1).contiguous()
    )
    with patch_load_sft_features(features, texts=_probe_texts(task, tokenizer)):
        generator = sft.make_sft_data_generator(
            task=task,
            tokenizer=tokenizer,
            B=1,
            T=sequence_len,
            device="cpu",
            sft_state=sft.SFTState(),
            feature_dir="/dummy/features",
            parquet_dir="/dummy/parquet",
            threshold_cfg=ThresholdConfig(ThresholdType.feature_percentile, 0.0),
            boundaries=boundaries,
            precomputed_thresholds=t.zeros(N_SAE, dtype=t.float32),
            min_loss_tokens=1,
            fp_percentile_table=_make_synthetic_percentile_table(N_SAE),
            fp_target_layout=layout,
            feature_mask=feature_mask,
            consolidation=consolidation,
            return_seg_doc_id=True,
        )
        return next(generator)


def test_precompaction_masked_fp_base_keeps_legacy_full_width_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_mask = FeatureMask("whitelist", (1, 4, 7))
    model_data = {
        FEATURE_EMBED_KEY: t.zeros((N_SAE, 4)),
        "feature_pred_head.weight": t.zeros((N_SAE, 4)),
        "feature_pred_head.bias": t.zeros(N_SAE),
    }
    meta = {
        "model_config": {
            "sequence_len": 16,
            "vocab_size": 128,
            "n_layer": 2,
            "n_head": 2,
            "n_kv_head": 2,
            "n_embd": 64,
            "window_pattern": "L",
        },
        "user_config": {
            "encoder": {
                "n_embd": 32,
                "depth": 1,
                "n_head": 2,
                "n_kv_head": 2,
                "head_dim": 16,
            },
            "cross_attention": {"frequency": 1, "n_head": 2, "n_kv_head": 2},
            "num_buckets": NUM_BUCKETS,
            "feature_prediction": {"block": "mlp", "head_layers": 0},
        },
        "feature_mask_canonical": canonical_dict(feature_mask),
    }
    cfg = OmegaConf.create(
        {
            "lr_warmup_steps": 0,
            "num_iterations": -1,
            "feature_clusters": None,
            "feature_prediction_block": "mlp",
            "feature_prediction_head_layers": 0,
            "sft_loss_mask": "full_sequence",
            "missing_percentile_policy": "error",
            "feature_mask": {"mode": "whitelist", "ids": [1, 4, 7]},
            "feature_selection": {
                "threshold_type": "feature_percentile",
                "threshold": 0.5,
            },
            "device_type": "cpu",
            "base_checkpoint_dir": str(tmp_path / "base"),
            "model_tag": "unit",
            "dataset": "safety",
            "resume_from_step": -1,
            "model_step": 1,
            "max_seq_len": None,
            "num_buckets": None,
            "run": {},
        }
    )
    layouts: list[FeaturePredictionTargetLayout] = []
    events: list[str] = []
    original_build = sft.FeaturePredictionTargetLayout.build

    def capture_layout(
        *,
        n_sae_features: PositiveInt,
        feature_mask: FeatureMask | None,
        feature_clusters: FeatureClusterConsolidation | None,
        feature_vocab: FeatureVocab | None,
    ) -> FeaturePredictionTargetLayout:
        layout = original_build(
            n_sae_features=n_sae_features,
            feature_mask=feature_mask,
            feature_clusters=feature_clusters,
            feature_vocab=feature_vocab,
        )
        layouts.append(layout)
        return layout

    monkeypatch.setattr(sft.sys, "argv", ["conditional_encdec_sft.py"])
    monkeypatch.setattr(sft, "initialize_config_dir", lambda **_: nullcontext())
    monkeypatch.setattr(sft, "compose", lambda **_: cfg)
    monkeypatch.setattr(sft, "compute_init", lambda _device: (False, 0, 0, 1, "cpu"))
    monkeypatch.setattr(sft, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        sft,
        "load_checkpoint",
        lambda *_args, **_kwargs: (model_data, None, meta),
    )
    monkeypatch.setattr(
        sft.FeaturePredictionTargetLayout,
        "build",
        staticmethod(capture_layout),
    )

    def reached_wandb(*_args, **_kwargs):
        events.append("create_wandb_run")
        raise RuntimeError("post-layout startup sentinel")

    monkeypatch.setattr(sft, "create_wandb_run", reached_wandb)

    with pytest.raises(RuntimeError, match="post-layout startup sentinel"):
        sft.main()

    assert events == ["create_wandb_run"]
    assert len(layouts) == 1
    layout = layouts[0]
    assert not layout.is_compact
    assert layout.feature_mask is None
    assert layout.feature_clusters is None
    assert layout.feature_vocab is None
    assert layout.output_width == N_SAE

    batch = _compact_batch(
        layout=layout,
        consolidation=None,
        feature_mask=feature_mask,
        segment_ids=((1, 2, 4),),
        segment_magnitudes=((71.0, 93.0, 22.0),),
    )
    fp_targets = batch[-2]
    assert fp_targets is not None
    assert fp_targets.shape[-1] == N_SAE
    target = fp_targets[0, 1]
    assert target[1].item() == pytest.approx(0.71)
    assert target[4].item() == pytest.approx(0.22)
    assert target[2].item() == 0.0


class TestCompactFpMainWiring:
    @pytest.fixture(scope="class")
    def main_source(self) -> str:
        return inspect.getsource(sft.main)

    def test_main_wires_resume_layout_validation(self, main_source: str) -> None:
        assert "_assert_resume_feature_prediction_layout_matches(" in main_source

    def test_main_writes_feature_prediction_layout_metadata(self, main_source: str) -> None:
        assert "_feature_prediction_layout_checkpoint_metadata(" in main_source


def test_mask_compacted_targets_map_original_ids_and_drop_masked_values() -> None:
    feature_mask = FeatureMask("whitelist", (1, 4, 7))
    layout, consolidation = _target_layout(feature_mask=feature_mask)
    assert layout.feature_vocab is not None

    batch = _compact_batch(
        layout=layout,
        consolidation=consolidation,
        feature_mask=feature_mask,
        segment_ids=((1, 2, 4),),
        segment_magnitudes=((71.0, 93.0, 22.0),),
    )
    fp_targets = batch[-2]
    assert fp_targets is not None
    assert fp_targets.shape[-1] == layout.output_width
    target = fp_targets[0, 1]
    expected_by_original = {1: 0.71, 4: 0.22, 7: 0.0}
    for row, original_id in enumerate(layout.feature_vocab.original_ids):
        assert target[row].item() == pytest.approx(expected_by_original[original_id])

    masked_original_percentile = t.tensor(0.93, dtype=fp_targets.dtype)
    assert not bool(t.isclose(fp_targets, masked_original_percentile, rtol=0.0, atol=1e-7).any())


def test_clustered_targets_max_reduce_siblings_not_sum_or_last_write(
    tmp_path: Path,
) -> None:
    resolved = _resolved_clusters(tmp_path, ((1, 3),))
    layout, consolidation = _target_layout(resolved_clusters=resolved)
    assert layout.feature_vocab is not None
    assert consolidation is not None

    batch = _compact_batch(
        layout=layout,
        consolidation=consolidation,
        feature_mask=None,
        segment_ids=((1, 3, 5),),
        segment_magnitudes=((71.0, 23.0, 41.0),),
    )
    fp_targets = batch[-2]
    assert fp_targets is not None
    representative_row = layout.feature_vocab.original_ids.index(1)
    representative_value = fp_targets[0, 1, representative_row].item()
    assert representative_value == pytest.approx(0.71)
    assert representative_value != pytest.approx(0.71 + 0.23)
    assert representative_value != pytest.approx(0.23)


def test_duplicate_original_id_in_one_sft_segment_is_corruption() -> None:
    feature_mask = FeatureMask("whitelist", (1, 5, 7))
    layout, consolidation = _target_layout(feature_mask=feature_mask)

    with pytest.raises(
        ValueError,
        match=r"example 0 segment 0 contains duplicate original SAE feature ID 1",
    ):
        _compact_batch(
            layout=layout,
            consolidation=consolidation,
            feature_mask=feature_mask,
            segment_ids=((1, 1, 5),),
            segment_magnitudes=((71.0, 23.0, 41.0),),
        )


@pytest.fixture
def clustered_compact_resume_contract(
    tmp_path: Path,
) -> tuple[
    FeaturePredictionTargetLayout,
    ResolvedFeatureClusters,
    FeatureMask,
    dict,
    str,
]:
    feature_mask = FeatureMask("whitelist", (1, 3, 5, 6))
    resolved = _resolved_clusters(tmp_path, ((1, 3),))
    layout, _consolidation = _target_layout(
        feature_mask=feature_mask,
        resolved_clusters=resolved,
    )
    assert layout.feature_vocab is not None
    meta = {
        "step": 1,
        "feature_mask_canonical": canonical_dict(feature_mask),
        "feature_vocab": layout.feature_vocab.metadata_dict(),
        **feature_clusters_checkpoint_metadata(resolved),
        **sft._feature_prediction_layout_checkpoint_metadata(layout),
    }
    checkpoint_dir = str(tmp_path / "sft")
    save_checkpoint(
        checkpoint_dir,
        step=1,
        model_data={"dummy": t.zeros(1)},
        optimizer_data=None,
        meta_data=meta,
    )
    _model, _optimizer, loaded_meta = load_checkpoint(checkpoint_dir, 1, "cpu")
    return layout, resolved, feature_mask, loaded_meta, checkpoint_dir


def test_compact_fp_resume_rejects_mismatched_cluster_descriptor(
    clustered_compact_resume_contract,
    tmp_path: Path,
) -> None:
    _layout, _resolved, _mask, meta, checkpoint_dir = clustered_compact_resume_contract
    mismatched = _resolved_clusters(tmp_path, ((1, 5),))
    with pytest.raises(FeatureClustersError, match="mismatch on resume"):
        assert_resume_feature_clusters_match(
            mismatched,
            meta,
            checkpoint_dir=checkpoint_dir,
        )


def test_compact_fp_resume_rejects_mismatched_vocab(
    clustered_compact_resume_contract,
) -> None:
    layout, _resolved, _mask, meta, checkpoint_dir = clustered_compact_resume_contract
    assert layout.feature_vocab is not None
    mismatched = FeatureVocab(
        n_sae_features=N_SAE,
        original_ids=layout.feature_vocab.original_ids[:-1],
    )
    with pytest.raises(FeatureVocabError, match="mismatch"):
        assert_resume_feature_vocab_matches(
            mismatched,
            meta,
            checkpoint_dir=checkpoint_dir,
        )


def test_compact_fp_resume_rejects_partial_metadata_without_width(
    clustered_compact_resume_contract,
) -> None:
    layout, _resolved, _mask, meta, checkpoint_dir = clustered_compact_resume_contract
    meta.pop("feature_prediction_width")
    with pytest.raises(
        ValueError,
        match=r"compact-FP metadata is missing or partial.*pre-#1276",
    ):
        sft._assert_resume_feature_prediction_layout_matches(
            layout,
            meta,
            checkpoint_dir=checkpoint_dir,
        )


def test_compact_fp_resume_rejects_mismatched_width(
    clustered_compact_resume_contract,
) -> None:
    layout, _resolved, _mask, meta, checkpoint_dir = clustered_compact_resume_contract
    meta["feature_prediction_width"] = layout.output_width + 1
    with pytest.raises(FeatureVocabError, match="feature-prediction width mismatch"):
        sft._assert_resume_feature_prediction_layout_matches(
            layout,
            meta,
            checkpoint_dir=checkpoint_dir,
        )


def test_old_full_width_fp_resume_may_omit_width() -> None:
    layout = FeaturePredictionTargetLayout.build(
        n_sae_features=PositiveInt.parse(N_SAE),
        feature_mask=None,
        feature_clusters=None,
        feature_vocab=None,
    )
    sft._assert_resume_feature_prediction_layout_matches(
        layout,
        {"step": 1},
        checkpoint_dir="/legacy-sft",
    )


def test_full_width_fp_resume_rejects_orphan_compact_vocab_metadata() -> None:
    layout = FeaturePredictionTargetLayout.build(
        n_sae_features=PositiveInt.parse(N_SAE),
        feature_mask=None,
        feature_clusters=None,
        feature_vocab=None,
    )
    orphan_vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=(1, 3, 5))
    with pytest.raises(ValueError, match="compact feature vocabulary.*full-width"):
        sft._assert_resume_feature_prediction_layout_matches(
            layout,
            {"step": 1, "feature_vocab": orphan_vocab.metadata_dict()},
            checkpoint_dir="/legacy-sft",
        )


def test_compact_fp_batch_runs_finite_forward_backward_with_all_gradients(
    tmp_path: Path,
) -> None:
    resolved = _resolved_clusters(tmp_path, ((1, 3),))
    layout, consolidation = _target_layout(resolved_clusters=resolved)
    assert layout.feature_vocab is not None
    assert consolidation is not None
    batch = _compact_batch(
        layout=layout,
        consolidation=consolidation,
        feature_mask=None,
        segment_ids=((1, 3, 5), (1, 3, 6)),
        segment_magnitudes=((71.0, 23.0, 41.0), (31.0, 83.0, 52.0)),
    )
    inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets, seg_doc_id = batch
    fp_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.FULL_SEGMENT,
        loss_fn=FeaturePredLossFn.BCE,
    )
    model = EncoderDecoderGPT(
        GPTConfig(
            sequence_len=inputs.shape[1],
            vocab_size=100,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            window_pattern="L",
        ),
        EncoderConfig(n_embd=32, depth=1, n_head=2, n_kv_head=2, head_dim=16),
        CrossAttentionConfig(frequency=1, n_head=2, n_kv_head=2),
        n_sae_features=N_SAE,
        num_buckets=NUM_BUCKETS,
        feature_pred_config=fp_config,
        feature_vocab=layout.feature_vocab,
    )
    model.init_weights()
    with t.no_grad():
        generator = t.Generator().manual_seed(1276)
        for layer in model.cross_attn_layers.values():
            c_proj = layer.c_proj
            assert isinstance(c_proj, t.nn.Linear)
            c_proj.weight.copy_(t.randn(c_proj.weight.shape, generator=generator) * 0.02)

    result = model(
        inputs,
        targets,
        enc_fids,
        enc_mags,
        enc_mask,
        seg_map,
        fp_targets=fp_targets,
        seg_doc_id=seg_doc_id,
    )
    assert isinstance(result, ForwardResult)
    assert t.isfinite(result.loss)
    result.loss.backward()

    cross_attention = next(iter(model.cross_attn_layers.values()))
    c_q = cross_attention.c_q
    assert isinstance(c_q, t.nn.Linear)
    assert model.feature_pred_head is not None
    gradient_groups = {
        "encoder": model.encoder.feature_embed.weight.grad,
        "cross_attention": c_q.weight.grad,
        "feature_pred_head": model.feature_pred_head.weight.grad,
    }
    for name, gradient in gradient_groups.items():
        assert gradient is not None, f"{name} gradient is missing"
        assert t.isfinite(gradient).all(), f"{name} gradient is not finite"
        assert gradient.abs().sum().item() > 0.0, f"{name} gradient is zero"
