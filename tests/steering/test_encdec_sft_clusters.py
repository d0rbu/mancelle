"""SFT cluster-contract tests and the pre-#1279 unclustered oracle."""

from __future__ import annotations

import ast
import inspect
import json
import math
import shutil
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

import nanochat.flash_attention as _fa_mod
from errors import FeatureClustersError, FeatureVocabError
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from scripts.train import conditional_encdec_sft as sft
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    assert_resume_feature_clusters_match,
    feature_clusters_checkpoint_metadata,
    reject_feature_clusters_metadata,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_vocab import (
    FEATURE_EMBED_KEY,
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
    FeatureVocab,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from tests.steering.goldens.gen_encdec_sft_unclustered_golden import (
    GOLDEN_PATH,
    ORACLE_COMMIT,
    build_capture,
)
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    _full_sequence_char_count,
    patch_load_sft_features,
)

REPO_ROOT = Path(__file__).parents[2]
SFT_CONFIG_DIR = REPO_ROOT / "configs" / "training"
SFT_CONFIGS = (
    "conditional_encdec_sft",
    "conditional_encdec_sft_4090",
    "conditional_encdec_sft_2xh200",
    "conditional_encdec_sft_4xh200",
    "conditional_encdec_sft_2xb200",
    "conditional_encdec_sft_profile_4090",
)
REMOVED_FREQ_KEYS = {"feature_mask_freq_min", "feature_mask_freq_max"}


def _resolved(
    tmp_path: Path,
    *,
    n_sae_features: int = 8,
    groups: tuple[tuple[int, ...], ...] = ((1, 3),),
) -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(clusters=groups)
    descriptor = FeatureClusterDescriptor.from_definition(
        definition,
        n_sae_features=n_sae_features,
    )
    return ResolvedFeatureClusters(
        source_path=tmp_path / "clusters.yaml",
        definition=definition,
        descriptor=descriptor,
    )


def _golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_unclustered_golden_was_captured_at_the_required_base() -> None:
    assert _golden()["oracle_commit"] == ORACLE_COMMIT


def test_unclustered_generator_matches_the_base_capture() -> None:
    golden = _golden()
    actual = build_capture()
    dtype_by_name = {
        "torch.int64": t.int64,
        "torch.bool": t.bool,
    }
    for name, expected in golden["tensors"].items():
        observed = actual["tensors"][name]
        assert observed["sha256"] == expected["sha256"], name
        assert t.equal(
            t.tensor(observed["values"], dtype=dtype_by_name[observed["dtype"]]),
            t.tensor(expected["values"], dtype=dtype_by_name[expected["dtype"]]),
        ), name


def test_unclustered_checkpoint_path_matches_the_base_capture() -> None:
    assert build_capture()["checkpoint_path"] == _golden()["checkpoint_path"]


def test_unclustered_leaf_composition_is_a_noop_and_matches_the_golden_path() -> None:
    base = "/checkpoints/d26/s128/t10"
    composed = sft.append_feature_clusters_leaf(sft.append_mask_leaf(base, None), None)

    assert composed == base
    assert sft._build_sft_checkpoint_dir(composed, "safety") == _golden()["checkpoint_path"]


def test_unclustered_checkpoint_user_config_keys_keep_only_the_intentional_delta() -> None:
    actual = set(build_capture()["checkpoint_user_config_keys"])
    expected = (
        set(_golden()["checkpoint_user_config_keys"])
        - REMOVED_FREQ_KEYS
        - set(sft._UNSUPPORTED_SFT_PRETRAIN_KEYS)
    )
    # Intentional #1286/#1285 deltas: the shipped loss-mask and alignment keys.
    # Issue #1281 removes the 18 pretraining-only keys above from the SFT surface.
    # Conditioning fidelity vs pretraining adds NO key: it is hardcoded policy
    # (steering/conditional/checkpoint_conditioning.assert_conditioning_supported),
    # so there is nothing per-run to record or audit.
    # #1311 adds the explicit missing-percentile policy to the recorded surface:
    # a `lowest` pilot must be auditable from the checkpoint alone. Issue #1475
    # adds bounded SFT batch prefetching; it is a recorded execution setting
    # but does not change conditioning. The steered SFT mixtures add
    # `steered_jsonl`: the ONE mixture-sizing override the trainer forwards to
    # `build_task_mixtures`, and therefore mixture identity that has to be
    # recoverable from the checkpoint alone (a different steered file is a
    # different row order and a different cursor). Issue #1719 records the head
    # block kind and depth explicitly in the SFT override surface.
    expected.update(
        {
            "sft_loss_mask",
            "alignment_probe_samples",
            "missing_percentile_policy",
            "sft_prefetch_batches",
            "steered_jsonl",
            "feature_prediction_block",
            "feature_prediction_head_layers",
            "fp_head_lr",
            "fp_head_matrix_lr",
            "lr_warmup_steps",
        }
    )
    assert actual == expected
    assert "feature_clusters" not in actual
    assert actual.isdisjoint(REMOVED_FREQ_KEYS)


@pytest.mark.parametrize("config_name", SFT_CONFIGS)
def test_all_sft_configs_expose_explicit_clusters_without_frequency_producer(
    config_name: str,
) -> None:
    with initialize_config_dir(config_dir=str(SFT_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name)

    assert cfg.feature_clusters is None
    assert REMOVED_FREQ_KEYS.isdisjoint(cfg)


@pytest.mark.parametrize("removed_key", sorted(REMOVED_FREQ_KEYS))
def test_removed_frequency_override_fails_in_hydra_struct_mode(removed_key: str) -> None:
    with pytest.raises(
        ConfigCompositionException,
        match=rf"Could not override '{removed_key}'",
    ):
        with initialize_config_dir(config_dir=str(SFT_CONFIG_DIR), version_base=None):
            compose(
                config_name="conditional_encdec_sft",
                overrides=[f"{removed_key}=0.1"],
            )


@pytest.mark.parametrize("removed_key", sorted(REMOVED_FREQ_KEYS))
def test_removed_frequency_append_override_hits_the_post_compose_guard(
    removed_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sft.sys,
        "argv",
        ["conditional_encdec_sft.py", f"+{removed_key}=0.125"],
    )

    with pytest.raises(
        ValueError,
        match=rf"{removed_key}.*exact explicit feature_mask.*dump_feature_mask",
    ):
        sft.main()


@pytest.mark.parametrize("removed_key", sorted(REMOVED_FREQ_KEYS))
def test_removed_frequency_inherited_config_hits_the_post_compose_guard(
    removed_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "training"
    shutil.copytree(SFT_CONFIG_DIR, config_dir)
    (config_dir / "conditional_encdec_sft_removed_frequency.yaml").write_text(
        f"defaults:\n  - conditional_encdec_sft\n  - _self_\n{removed_key}: 0.125\n",
        encoding="utf-8",
    )
    initialize = sft.initialize_config_dir
    monkeypatch.setattr(
        sft,
        "initialize_config_dir",
        lambda **_: initialize(config_dir=str(config_dir), version_base=None),
    )
    monkeypatch.setattr(
        sft.sys,
        "argv",
        ["conditional_encdec_sft.py", "config_name=conditional_encdec_sft_removed_frequency"],
    )

    with pytest.raises(
        ValueError,
        match=rf"{removed_key}.*exact explicit feature_mask.*dump_feature_mask",
    ):
        sft.main()


def test_frequency_producer_is_absent_from_sft_source_and_imports() -> None:
    source = Path(sft.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "maybe_apply_frequency_blacklist" not in source
    assert not any(
        isinstance(node, ast.Name) and node.id == "maybe_apply_frequency_blacklist"
        for node in ast.walk(tree)
    )


class TestConditionalClusterGates:
    def test_unclustered_run_rejects_clustered_base_metadata(self, tmp_path: Path) -> None:
        resolved = _resolved(tmp_path)
        meta = feature_clusters_checkpoint_metadata(resolved)

        with pytest.raises(FeatureClustersError, match="base checkpoint.*cannot consume"):
            reject_feature_clusters_metadata(
                meta,
                "conditional_encdec_sft base checkpoint /base (run has feature_clusters disabled)",
            )

    def test_clustered_run_rejects_unclustered_base_metadata(self, tmp_path: Path) -> None:
        with pytest.raises(FeatureClustersError, match="lacks complete"):
            assert_resume_feature_clusters_match(
                _resolved(tmp_path),
                {},
                checkpoint_dir="/base",
            )

    def test_agreeing_clustered_base_reaches_the_vocab_guard(self, tmp_path: Path) -> None:
        resolved = _resolved(tmp_path)
        assert_resume_feature_clusters_match(
            resolved,
            feature_clusters_checkpoint_metadata(resolved),
            checkpoint_dir="/base",
        )
        consolidation = FeatureClusterConsolidation.build(resolved, None)
        wrong_vocab = FeatureVocab(n_sae_features=8, original_ids=(0, 1, 2))

        with pytest.raises(FeatureVocabError, match="feature_clusters descriptor"):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                consolidation,
                None,
                wrong_vocab,
                8,
                checkpoint_dir="/base",
                base_meta={},
            )

    def test_resume_with_mismatched_cluster_descriptor_raises(self, tmp_path: Path) -> None:
        configured = _resolved(tmp_path, groups=((1, 3),))
        checkpoint = _resolved(tmp_path, groups=((2, 4),))

        with pytest.raises(FeatureClustersError, match="mismatch on resume"):
            assert_resume_feature_clusters_match(
                configured,
                feature_clusters_checkpoint_metadata(checkpoint),
                checkpoint_dir="/sft",
            )

    def test_unclustered_resume_rejects_clustered_sft_checkpoint(self, tmp_path: Path) -> None:
        resolved = _resolved(tmp_path)

        with pytest.raises(FeatureClustersError, match="run has feature_clusters disabled"):
            assert_resume_feature_clusters_match(
                None,
                feature_clusters_checkpoint_metadata(resolved),
                checkpoint_dir="/sft",
            )


def test_clustered_raw_magnitude_threshold_rejected_at_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "dataset": "safety",
            "lr_warmup_steps": 0,
            "num_iterations": -1,
            "feature_prediction_block": "mlp",
            "feature_prediction_head_layers": 0,
            "sft_loss_mask": "full_sequence",
            "missing_percentile_policy": "error",
            "feature_clusters": {"path": str(tmp_path / "clusters.yaml")},
            "feature_selection": {
                "threshold_type": "segment_topk",
                "threshold": 1,
            },
        }
    )
    monkeypatch.setattr(sft.sys, "argv", ["conditional_encdec_sft.py"])
    monkeypatch.setattr(sft, "initialize_config_dir", lambda **_: nullcontext())
    monkeypatch.setattr(sft, "compose", lambda **_: cfg)

    with pytest.raises(
        FeatureClustersError,
        match=r"requires raw feature magnitudes.*only has percentiles.*Supported types",
    ):
        sft.main()


def test_clustered_percentile_segment_topk_reaches_compute_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "dataset": "safety",
            "lr_warmup_steps": 0,
            "num_iterations": -1,
            "feature_prediction_block": "mlp",
            "feature_prediction_head_layers": 0,
            "sft_loss_mask": "full_sequence",
            "missing_percentile_policy": "error",
            "feature_clusters": {"path": str(tmp_path / "clusters.yaml")},
            "feature_selection": {
                "threshold_type": "feature_percentile_segment_topk",
                "threshold": 1.4,
            },
            "device_type": "cpu",
        }
    )
    events: list[str] = []
    monkeypatch.setattr(sft.sys, "argv", ["conditional_encdec_sft.py"])
    monkeypatch.setattr(sft, "initialize_config_dir", lambda **_: nullcontext())
    monkeypatch.setattr(sft, "compose", lambda **_: cfg)

    def reached_compute_init(_device: str) -> None:
        events.append("compute_init")
        raise RuntimeError("compute_init sentinel after clustered percentile guard")

    monkeypatch.setattr(sft, "compute_init", reached_compute_init)

    with pytest.raises(RuntimeError, match="compute_init sentinel"):
        sft.main()

    assert events == ["compute_init"]


def test_clustered_compact_fp_checkpoint_reaches_post_contract_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolved(tmp_path)
    vocab_ids = (0, 1, 2, 4, 5, 6, 7)
    model_data = {
        FEATURE_EMBED_KEY: t.zeros((len(vocab_ids), 4)),
        VOCAB_IDS_KEY: t.tensor(vocab_ids, dtype=t.int64),
        VOCAB_N_SAE_KEY: t.tensor(8, dtype=t.int64),
        "feature_pred_head.weight": t.zeros((len(vocab_ids), 4)),
        "feature_pred_head.bias": t.zeros(len(vocab_ids)),
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
            "num_buckets": 4,
            "feature_clusters": {"path": str(resolved.source_path)},
            "feature_prediction": {"block": "mlp", "head_layers": 0},
        },
        **feature_clusters_checkpoint_metadata(resolved),
    }
    cfg = OmegaConf.create(
        {
            "dataset": "safety",
            "lr_warmup_steps": 0,
            "num_iterations": -1,
            "feature_prediction_block": "mlp",
            "feature_prediction_head_layers": 0,
            "sft_loss_mask": "full_sequence",
            "missing_percentile_policy": "error",
            "feature_clusters": {"path": str(resolved.source_path)},
            "feature_selection": {
                "threshold_type": "feature_percentile",
                "threshold": 0.5,
            },
            "device_type": "cpu",
            "base_checkpoint_dir": str(tmp_path / "base"),
            "model_tag": "unit",
            "resume_from_step": -1,
            "model_step": 1,
            "max_seq_len": None,
            "num_buckets": None,
            "run": {},
        }
    )
    events: list[str] = []
    monkeypatch.setattr(sft.sys, "argv", ["conditional_encdec_sft.py"])
    monkeypatch.setattr(sft, "initialize_config_dir", lambda **_: nullcontext())
    monkeypatch.setattr(sft, "compose", lambda **_: cfg)
    monkeypatch.setattr(sft, "compute_init", lambda _device: (False, 0, 0, 1, "cpu"))
    monkeypatch.setattr(sft, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sft, "resolve_feature_clusters", lambda *_args: resolved)
    monkeypatch.setattr(
        sft,
        "load_checkpoint",
        lambda *_args, **_kwargs: (model_data, None, meta),
    )

    def reached_wandb(*_args, **_kwargs):
        events.append("create_wandb_run")
        raise RuntimeError("post-contract startup sentinel")

    monkeypatch.setattr(sft, "create_wandb_run", reached_wandb)

    with pytest.raises(RuntimeError, match="post-contract startup sentinel"):
        sft.main()

    assert events == ["create_wandb_run"]


def test_unclustered_main_resume_rejects_clustered_sft_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolved(tmp_path)
    model_data = {FEATURE_EMBED_KEY: t.zeros((8, 4))}
    base_meta = {
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
            "num_buckets": 4,
        },
    }
    cfg = OmegaConf.create(
        {
            "feature_clusters": None,
            "lr_warmup_steps": 0,
            "num_iterations": -1,
            "feature_prediction_block": "mlp",
            "feature_prediction_head_layers": 0,
            "sft_loss_mask": "full_sequence",
            "missing_percentile_policy": "error",
            "feature_mask": None,
            "feature_selection": {
                "threshold_type": "segment_topk",
                "threshold": 1,
            },
            "device_type": "cpu",
            "base_checkpoint_dir": str(tmp_path / "base"),
            "model_tag": "unit",
            "dataset": "safety",
            "resume_from_step": 1,
            "model_step": 1,
            "max_seq_len": None,
            "num_buckets": None,
        }
    )
    load_count = 0

    def load_checkpoint(*_args, **_kwargs):
        nonlocal load_count
        load_count += 1
        if load_count == 1:
            return model_data, None, base_meta
        return model_data, {}, feature_clusters_checkpoint_metadata(resolved)

    monkeypatch.setattr(sft.sys, "argv", ["conditional_encdec_sft.py"])
    monkeypatch.setattr(sft, "initialize_config_dir", lambda **_: nullcontext())
    monkeypatch.setattr(sft, "compose", lambda **_: cfg)
    monkeypatch.setattr(sft, "compute_init", lambda _device: (False, 0, 0, 1, "cpu"))
    monkeypatch.setattr(sft, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sft, "load_checkpoint", load_checkpoint)

    with pytest.raises(
        FeatureClustersError,
        match=r"resume checkpoint.*run has feature_clusters disabled.*cannot consume",
    ):
        sft.main()

    assert load_count == 2


def _clustered_generator_batch(
    tmp_path: Path,
    *,
    threshold_cfg: ThresholdConfig | None = None,
    sampling_bundle: SamplingBundle | None = None,
    boundaries: t.Tensor | None = None,
    supply_cluster_threshold_table: bool = True,
) -> tuple[tuple, FeatureClusterConsolidation, int]:
    assert (threshold_cfg is None) != (sampling_bundle is None)
    n_sae_features = 8
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=4, user_chars=3, asst_chars=7, prefix="c")
    total_chars = _full_sequence_char_count(tokenizer, task[0])
    raw_segment = SegmentFeatures(
        segment_idx=0,
        position=0,
        feature_ids=t.tensor([1, 3, 5], dtype=t.int32),
        feature_mags=t.tensor([0.40, 0.45, 0.60], dtype=t.float32),
        char_start=0,
        char_end=total_chars,
    )
    features = {index: [raw_segment] for index in range(len(task))}
    resolved = _resolved(tmp_path, n_sae_features=n_sae_features)
    consolidation = FeatureClusterConsolidation.build(resolved, None)
    if boundaries is None:
        boundaries = (
            t.tensor([0.25, 0.50, 0.75], dtype=t.float64)
            .unsqueeze(0)
            .expand(n_sae_features, -1)
            .contiguous()
        )
    full_percentile_table = (
        t.linspace(0.01, 0.99, 99, dtype=t.float64)
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .contiguous()
    )
    rendered, _ = tokenizer.render_conversation(task[0])
    sequence_len = len(rendered) - 1
    with patch_load_sft_features(features):
        generator = sft.make_sft_data_generator(
            task=task,
            tokenizer=tokenizer,
            B=2,
            T=sequence_len,
            device="cpu",
            sft_state=sft.SFTState(),
            feature_dir="/dummy/features",
            parquet_dir="/dummy/parquet",
            threshold_cfg=threshold_cfg,
            boundaries=boundaries,
            sampling_bundle=sampling_bundle,
            cluster_threshold_percentile_table=(
                full_percentile_table
                if threshold_cfg is not None and supply_cluster_threshold_table
                else None
            ),
            min_loss_tokens=1,
            consolidation=consolidation,
            alignment_probe_samples=0,
        )
        batch = next(generator)
    return batch, consolidation, sequence_len


def _sampling_bundle(mode: SamplingMode, *, min_k: int, max_k: int) -> SamplingBundle:
    percentile_table = (
        t.linspace(0.01, 0.99, 99, dtype=t.float64).unsqueeze(0).expand(8, -1).contiguous()
    )
    return SamplingBundle(
        selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=min_k,
            max_k=max_k,
            mode=mode,
        ),
        percentile_table=percentile_table,
        rng=t.Generator().manual_seed(1279),
    )


def test_clustered_top_k_percentile_sampling_folds_and_uses_normalized_buckets(
    tmp_path: Path,
) -> None:
    batch, _consolidation, _sequence_len = _clustered_generator_batch(
        tmp_path,
        sampling_bundle=_sampling_bundle(
            SamplingMode.top_k_percentile,
            min_k=1,
            max_k=2,
        ),
    )
    _inputs, _targets, enc_fids, enc_mags, enc_mask, _seg_map, _fp_targets = batch

    assert enc_fids[enc_mask].tolist() == [5, 5]
    assert enc_mags[enc_mask].tolist() == [2, 2]
    assert not bool((enc_fids == 3).any())


def test_clustered_weighted_sampling_folds_to_hand_computed_representative_buckets(
    tmp_path: Path,
) -> None:
    batch, _consolidation, _sequence_len = _clustered_generator_batch(
        tmp_path,
        sampling_bundle=_sampling_bundle(
            SamplingMode.weighted,
            min_k=2,
            max_k=3,
        ),
    )
    _inputs, _targets, enc_fids, enc_mags, enc_mask, _seg_map, _fp_targets = batch

    for row in range(enc_fids.shape[0]):
        row_ids = enc_fids[row][enc_mask[row]].tolist()
        row_buckets = enc_mags[row][enc_mask[row]].tolist()
        assert set(row_ids) == {1, 5}
        assert dict(zip(row_ids, row_buckets, strict=True)) == {1: 1, 5: 2}
    assert not bool((enc_fids == 3).any())


def test_clustered_feature_percentile_threshold_has_hand_computed_fold(
    tmp_path: Path,
) -> None:
    batch, _consolidation, _sequence_len = _clustered_generator_batch(
        tmp_path,
        threshold_cfg=ThresholdConfig(ThresholdType.feature_percentile, 0.6),
    )
    _inputs, _targets, enc_fids, enc_mags, enc_mask, _seg_map, _fp_targets = batch

    assert enc_fids[enc_mask].tolist() == [5, 5]
    assert enc_mags[enc_mask].tolist() == [2, 2]
    assert not bool((enc_fids == 3).any())


def test_clustered_threshold_uses_full_rank_not_post_threshold_boundary_rank(
    tmp_path: Path,
) -> None:
    """A full-rank p60 value stays selected over a p60-p99 bucket grid."""
    n_sae_features = 8
    post_threshold_boundaries = (
        t.tensor([0.73, 0.86, 0.99], dtype=t.float64)
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .contiguous()
    )
    batch, _consolidation, _sequence_len = _clustered_generator_batch(
        tmp_path,
        threshold_cfg=ThresholdConfig(ThresholdType.feature_percentile, 0.6),
        boundaries=post_threshold_boundaries,
    )
    _inputs, _targets, enc_fids, enc_mags, enc_mask, _seg_map, _fp_targets = batch

    assert enc_fids[enc_mask].tolist() == [5, 5]
    assert enc_mags[enc_mask].tolist() == [2, 2]


def _main_ast_tree() -> ast.AST:
    return ast.parse(inspect.getsource(sft.main))


def _clustered_threshold_predicate_blocks(tree: ast.AST) -> list[ast.If]:
    """Every ``if consolidation is not None and threshold_cfg is not None:`` in main()."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "consolidation is not None and threshold_cfg is not None"
    ]


def _contains_call_named(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(sub, ast.Call)
        and (
            (isinstance(sub.func, ast.Name) and sub.func.id == name)
            or (isinstance(sub.func, ast.Attribute) and sub.func.attr == name)
        )
        for sub in ast.walk(node)
    )


class TestClusteredThresholdSemanticsWiring:
    """Static wiring guard for PR #1321 review finding C1 (resume-semantics stamp).

    Driving ``main()`` all the way to the resume-guard call site would require
    mocking dataset-mixture resolution, real tokenizer loading, and full model
    construction/state-dict loading -- the same reason
    ``test_clustered_compact_fp_checkpoint_reaches_post_contract_startup``
    above stops at ``create_wandb_run`` instead of going further. Like
    ``TestSFTMainResumeWiring`` in ``tests/nanochat/test_encdec_sft_resume.py``,
    this does not execute ``main()``; it parses its source and checks that the
    save-side stamp and the resume-side guard are BOTH still wired to the same
    predicate that decides "this run is clustered + threshold-selected"
    (``consolidation is not None and threshold_cfg is not None``, the exact
    predicate ``cluster_threshold_percentile_table`` itself is built under),
    and that the resume guard is further nested under ``if resuming:`` so it
    is never called for a fresh run.
    """

    @pytest.fixture(scope="class")
    def clustered_threshold_blocks(self) -> list[ast.If]:
        blocks = _clustered_threshold_predicate_blocks(_main_ast_tree())
        assert len(blocks) == 2, (
            "expected exactly two `if consolidation is not None and threshold_cfg "
            "is not None:` blocks in main() (one stamping the checkpoint save "
            f"meta, one guarding resume); found {len(blocks)}. If this changed "
            "on purpose, update this test to match the new wiring."
        )
        return blocks

    def test_resume_guard_is_called_under_resuming_inside_the_predicate(
        self, clustered_threshold_blocks: list[ast.If]
    ) -> None:
        guarded = [
            block
            for block in clustered_threshold_blocks
            if any(
                isinstance(inner, ast.If)
                and ast.unparse(inner.test) == "resuming"
                and _contains_call_named(inner, "assert_resume_clustered_threshold_semantics_match")
                for inner in block.body
            )
        ]
        assert len(guarded) == 1, (
            "assert_resume_clustered_threshold_semantics_match must be called "
            "exactly once, nested under `if resuming:` inside the clustered-"
            "threshold predicate block -- a pre-#1321 checkpoint must never "
            "resume cleanly into full-distribution-rank selection"
        )

    def test_save_meta_stamps_the_semantics_key_under_the_same_predicate(
        self, clustered_threshold_blocks: list[ast.If]
    ) -> None:
        stamped = [
            block
            for block in clustered_threshold_blocks
            if "CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY" in ast.unparse(block)
            and "ClusteredThresholdSelectionSemantics.FULL_DISTRIBUTION_RANK" in ast.unparse(block)
        ]
        assert len(stamped) == 1, (
            "the CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY stamp must be "
            "written exactly once, guarded by the clustered-threshold predicate -- "
            "sampling and unclustered runs must never stamp it"
        )

    def test_the_two_predicate_blocks_split_save_from_resume(
        self, clustered_threshold_blocks: list[ast.If]
    ) -> None:
        """Guard against one block accidentally doing both jobs (or neither)."""
        shapes = sorted(
            (
                _contains_call_named(block, "assert_resume_clustered_threshold_semantics_match"),
                "CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY" in ast.unparse(block),
            )
            for block in clustered_threshold_blocks
        )
        assert shapes == [(False, True), (True, False)], shapes


def test_clustered_threshold_requires_a_full_percentile_table(tmp_path: Path) -> None:
    with pytest.raises(
        FeatureClustersError,
        match="clustered threshold SFT requires a full-distribution percentile table",
    ):
        _clustered_generator_batch(
            tmp_path,
            threshold_cfg=ThresholdConfig(ThresholdType.feature_percentile, 0.6),
            supply_cluster_threshold_table=False,
        )


def test_clustered_percentile_threshold_topk_ranks_when_k_is_less_than_n(
    tmp_path: Path,
) -> None:
    batch, _consolidation, _sequence_len = _clustered_generator_batch(
        tmp_path,
        threshold_cfg=ThresholdConfig(
            ThresholdType.feature_percentile_segment_topk,
            1.4,
        ),
    )
    _inputs, _targets, enc_fids, enc_mags, enc_mask, _seg_map, _fp_targets = batch

    assert enc_fids[enc_mask].tolist() == [5, 5]
    assert enc_mags[enc_mask].tolist() == [2, 2]


def test_assert_representatives_only_rejects_a_non_representative_id(
    tmp_path: Path,
) -> None:
    consolidation = FeatureClusterConsolidation.build(
        _resolved(tmp_path, n_sae_features=8),
        None,
    )
    assert int(consolidation.rep_of[3].item()) == 1

    with pytest.raises(
        FeatureClustersError,
        match=r"unit selection contains consolidated-away feature ID 3",
    ):
        consolidation.assert_representatives_only(
            t.tensor([3], dtype=t.int64),
            surface="unit selection",
        )


def test_clustered_generator_batch_runs_through_compacted_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, consolidation, sequence_len = _clustered_generator_batch(
        tmp_path,
        threshold_cfg=ThresholdConfig(ThresholdType.feature_percentile, 0.4),
    )
    inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch
    vocab = effective_feature_vocab(consolidation, 8)
    assert vocab is not None and vocab.original_ids == (0, 1, 2, 4, 5, 6, 7)
    monkeypatch.setattr(_fa_mod, "_override_impl", FlashAttnImpl.SDPA)
    model = EncoderDecoderGPT(
        GPTConfig(
            sequence_len=sequence_len,
            vocab_size=128,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            window_pattern="L",
        ),
        EncoderConfig(n_embd=32, depth=1, n_head=2, n_kv_head=2, head_dim=16),
        CrossAttentionConfig(frequency=1, n_head=2, n_kv_head=2),
        n_sae_features=8,
        num_buckets=4,
        feature_vocab=vocab,
    )
    model.init_weights()

    result = model(inputs, targets, enc_fids, enc_mags, enc_mask, seg_map)
    loss = result.loss if hasattr(result, "loss") else result
    assert loss is not None
    assert math.isfinite(float(loss.item()))
