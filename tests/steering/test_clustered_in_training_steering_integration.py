"""High-level clustered in-training steering integration coverage.

This fixture crosses the production boundaries that isolated cache, generation,
probe, and cadence tests do not cross together:

* real packed-row, selection, and provenance artifacts on disk;
* startup-time clustered cache validation and the two-pass provenance join;
* the trainer's periodic steering helper;
* original representative IDs remapped by a compact encoder;
* real encoder and decoder forward passes for both conditioning arms;
* clustered representative/member probing and metric aggregation; and
* restoration of RNG, model mode, and an offloaded probe model.

Autoregressive decoding is replaced with one deterministic token per condition.
The expensive and bug-prone conditioning/model/probe boundaries still execute,
while the default CPU suite remains fast and deterministic.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
from omegaconf import OmegaConf

import nanochat.flash_attention as flash_attention_module
import scripts.train.conditional_encdec_train as train_module
import steering.eval.inline_steering_eval as inline_eval_module
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from scripts.train.conditional_encdec_train import (
    SharedState,
    _maybe_run_inline_steering_eval,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_mask import (
    FeatureMask,
    resolve_feature_mask,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.clustered_inline_cache import (
    CLUSTERED_INLINE_CACHE_SEMANTICS,
    prepare_clustered_inline_eval_cache,
)
from steering.eval.inline_steering_eval import (
    CLUSTERED_PROBE_SEMANTICS,
    InlineEvalCache,
    SteeringEvalConfig,
)
from steering.overlay_io import read_manifest
from tests.steering._splice_cf_cache_fixtures import (
    SELECTION_N_SAE,
    build_interleaved_env,
    run_selection_main,
    write_cluster_yaml,
)

_WHITELIST_MASK = {"mode": "whitelist", "ids": list(range(200))}
_MODEL_MAX_SEQ_LEN = 32
_NUM_BUCKETS = 2
_PROBE_DIM = 8


@dataclass
class _SharedState:
    feature_index: FeatureIndex | None
    steering_probe_model: Any
    steering_eval_cfg: SteeringEvalConfig


class _RecordingWandbRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, Any], int]] = []

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.logged.append((metrics, step))


class _ProbeSaeConfig:
    apply_b_dec_to_input = True
    activation_fn = "relu"


class _ProbeSae(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        feature_scale = t.linspace(0.01, 0.02, SELECTION_N_SAE)
        self.W_enc = t.nn.Parameter(
            feature_scale.unsqueeze(0).repeat(_PROBE_DIM, 1),
            requires_grad=False,
        )
        self.b_enc = t.nn.Parameter(
            t.zeros(SELECTION_N_SAE),
            requires_grad=False,
        )
        self.b_dec = t.nn.Parameter(
            t.zeros(_PROBE_DIM),
            requires_grad=False,
        )
        self.cfg = _ProbeSaeConfig()


class _RecordingProbeModel:
    def __init__(self, tokenizer: Any) -> None:
        self.model = t.nn.Linear(1, 1, bias=False)
        self.sae = _ProbeSae()
        self.tokenizer = tokenizer
        self.hook_name = "blocks.0.hook_resid_post"
        self.layer = 0
        self.nnterp_model = None
        self.activation_type = "residual"
        self.apply_b_dec = True
        self.to_calls: list[str] = []

    @property
    def device(self) -> t.device:
        return next(self.model.parameters()).device

    def to(self, device: t.device | str) -> _RecordingProbeModel:
        self.to_calls.append(str(device))
        self.model.to(device)
        self.sae.to(device)
        return self


def _cluster_state(
    cluster_path: str,
) -> tuple[FeatureMask | None, FeatureClusterConsolidation, FeatureVocab]:
    cfg = OmegaConf.create(
        {
            "feature_mask": _WHITELIST_MASK,
            "feature_clusters": {"path": cluster_path},
        }
    )
    feature_mask = resolve_feature_mask(cfg, SELECTION_N_SAE)
    resolved = resolve_feature_clusters(cfg, SELECTION_N_SAE)
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, feature_mask)
    feature_vocab = effective_feature_vocab(clusters, SELECTION_N_SAE)
    assert feature_vocab is not None
    return feature_mask, clusters, feature_vocab


def _build_compact_model(feature_vocab: FeatureVocab) -> EncoderDecoderGPT:
    dec_config = GPTConfig(
        sequence_len=_MODEL_MAX_SEQ_LEN,
        vocab_size=1_000,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    enc_config = EncoderConfig(
        n_embd=32,
        depth=1,
        n_head=2,
        n_kv_head=2,
        head_dim=16,
    )
    model = EncoderDecoderGPT(
        dec_config=dec_config,
        enc_config=enc_config,
        xattn_config=CrossAttentionConfig(
            frequency=2,
            n_head=2,
            n_kv_head=2,
        ),
        n_sae_features=SELECTION_N_SAE,
        num_buckets=_NUM_BUCKETS,
        feature_vocab=feature_vocab,
        pad_vocab_size_to=1_000,
    )
    t.manual_seed(123)
    model.init_weights()
    model.train()
    return model


def test_real_clustered_selection_drives_in_training_steering_end_to_end(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        flash_attention_module,
        "_override_impl",
        FlashAttnImpl.SDPA,
    )
    env = build_interleaved_env(tmp_path, n_shards=3)
    cluster_path = write_cluster_yaml(tmp_path)
    run_selection_main(
        env,
        mask=_WHITELIST_MASK,
        clusters_path=cluster_path,
        num_workers=1,
        splice_batch_size=4,
        monkeypatch=monkeypatch,
    )

    selection_dirs = sorted((env["packed_row_dir"] / "sel").iterdir())
    assert len(selection_dirs) == 1
    selection_dir = selection_dirs[0]
    selection_manifest = read_manifest(selection_dir)
    assert selection_manifest is not None
    splice_metadata = selection_manifest["metadata"]["splice_counterfactual"]

    feature_mask, clusters, feature_vocab = _cluster_state(cluster_path)
    feature_index = FeatureIndex(
        env["feature_dir"],
        percentile_checksum=env["pct_checksum"],
        tokenizer_checksum=env["tok_checksum"],
    )
    validation_data_dir = tmp_path / "validation_data"
    validation_data_dir.mkdir()
    shutil.copy2(
        Path(env["data_dir"]) / "shard_00001.parquet",
        validation_data_dir / "shard_00001.parquet",
    )

    clustered_build = prepare_clustered_inline_eval_cache(
        feature_dir=Path(env["feature_dir"]),
        packed_row_dir=env["packed_row_dir"],
        selection_dir=selection_dir,
        selection_hash=selection_dir.name,
        feature_index=feature_index,
        data_dir=str(validation_data_dir),
        n_eval_docs=2,
        seed=42,
        model_max_seq_len=_MODEL_MAX_SEQ_LEN,
        effective_buckets=1,
        feature_mask=feature_mask,
        feature_clusters=clusters,
        feature_vocab=feature_vocab,
        delta_config={
            "mode": "splice_counterfactual",
            "neg_delta_ratio": 1.0,
        },
        boundary_fingerprint=splice_metadata["boundary_fingerprint"],
        splice_batch_size=4,
    )
    assert len(clustered_build.selected_docs) == 2
    assert len(clustered_build.payload.targets) == 2
    assert (
        clustered_build.payload.source_identity["semantics"]
        == CLUSTERED_INLINE_CACHE_SEMANTICS
    )
    for selected, cached_target in zip(
        clustered_build.selected_docs,
        clustered_build.payload.targets,
        strict=True,
    ):
        assert cached_target.key == selected
        assert cached_target.donor_doc_idx != cached_target.doc_idx
    target_index, target = next(
        (index, cached_target)
        for index, cached_target in enumerate(clustered_build.payload.targets)
        if any(len(group) > 1 for group in cached_target.positive_member_groups)
    )
    prefix_feature_width = max(
        len(segment.feature_ids) for segment in target.prefix_segments
    )
    assert len(target.donor_segment.feature_ids) > prefix_feature_width

    inline_cache = InlineEvalCache(
        selected_docs=clustered_build.selected_docs,
        selection_metrics=clustered_build.selection_metrics,
        doc_texts=clustered_build.doc_texts,
        clustered_payload=clustered_build.payload,
    )
    model = _build_compact_model(feature_vocab)
    compact_rows_seen: list[int] = []

    def _capture_compact_rows(_module, args) -> None:
        compact_rows_seen.extend(int(row) for row in args[0].detach().flatten())

    model.encoder.feature_embed.register_forward_pre_hook(_capture_compact_rows)
    generated_inputs: list[Any] = []
    forward_logits: list[t.Tensor] = []

    def _one_token_generation(**kwargs):
        encoder_inputs = kwargs["encoder_inputs"]
        generated_inputs.append(encoder_inputs)
        device = kwargs["device"]
        assert isinstance(device, t.device)
        enc_feature_ids = encoder_inputs.enc_feature_ids.to(device)
        enc_magnitudes = encoder_inputs.enc_magnitudes.to(device)
        enc_mask = encoder_inputs.enc_mask.to(device)
        encoder_out = kwargs["model"].encoder(
            enc_feature_ids,
            enc_magnitudes,
            enc_mask,
        )
        prefix_tokens = kwargs["prefix_tokens"]
        prefix = t.tensor([prefix_tokens], dtype=t.int64, device=device)
        logits = kwargs["model"](
            prefix,
            targets=None,
            encoder_out=encoder_out,
            seg_map=encoder_inputs.seg_map[:, : len(prefix_tokens)].to(device),
            encoder_strict=True,
        )
        assert bool(t.isfinite(logits).all())
        forward_logits.append(logits.detach())
        # Prove the trainer restores RNG around steering evaluation.
        t.rand(1)
        return [[ord("z")]]

    monkeypatch.setattr(
        inline_eval_module,
        "generate_samples_for_condition",
        _one_token_generation,
    )
    monkeypatch.setattr(
        inline_eval_module,
        "get_probe_activations",
        lambda _model, tokens, **_kwargs: t.ones(
            tokens.shape[0],
            tokens.shape[1],
            _PROBE_DIM,
            device=tokens.device,
        ),
    )
    monkeypatch.setattr(
        train_module,
        "prepare_inline_eval_cache",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "clustered steering must reuse its startup-built cache"
            )
        ),
    )

    probe_model = _RecordingProbeModel(env["tokenizer"])
    shared = _SharedState(
        feature_index=None,
        steering_probe_model=probe_model,
        steering_eval_cfg=SteeringEvalConfig(
            n_eval_docs=2,
            samples_per_condition=1,
            seed=42,
        ),
    )
    cfg = OmegaConf.create(
        {
            "steering_eval_every": 1,
            "offload_steering_eval_feature_model": True,
            "data_dir": str(validation_data_dir),
        }
    )
    wandb_run = _RecordingWandbRun()

    percentile_table = t.linspace(
        -2.0,
        2.0,
        99,
        dtype=t.float64,
    ).repeat(SELECTION_N_SAE, 1)
    t.manual_seed(987)
    rng_before = t.random.get_rng_state().clone()
    returned_cache = _maybe_run_inline_steering_eval(
        step=1,
        last_step=False,
        cfg=cfg,
        shared=cast(SharedState, shared),
        model=model,
        orig_model=model,
        feature_dir=env["feature_dir"],
        steering_threshold_cfg=None,
        steering_precomputed_thresholds=None,
        steering_percentile_table=percentile_table,
        tokenizer=env["tokenizer"],
        device=t.device("cpu"),
        device_type="cpu",
        boundaries=t.empty(0),
        ddp_rank=0,
        ddp_world_size=1,
        master_process=True,
        wandb_run=wandb_run,
        inline_eval_cache=inline_cache,
        feature_mask=feature_mask,
    )

    assert returned_cache is inline_cache
    assert model.training
    assert t.equal(t.random.get_rng_state(), rng_before)
    assert probe_model.to_calls == ["cpu", "cpu"]
    assert len(generated_inputs) == 4
    assert len(forward_logits) == 4

    real_inputs, random_inputs = generated_inputs[
        2 * target_index : 2 * target_index + 2
    ]
    target_slot = target.target_segment_ordinal
    real_ids = real_inputs.enc_feature_ids[0, target_slot][
        real_inputs.enc_mask[0, target_slot]
    ].tolist()
    random_ids = random_inputs.enc_feature_ids[0, target_slot][
        random_inputs.enc_mask[0, target_slot]
    ].tolist()
    assert real_ids == target.prefix_segments[target_slot].feature_ids.tolist()
    assert random_ids == target.donor_segment.feature_ids.tolist()
    assert real_ids != random_ids

    all_original_ids = t.cat(
        [
            inputs.enc_feature_ids[inputs.enc_mask]
            for inputs in generated_inputs
        ]
    ).to(t.int64)
    expected_compact_rows = set(feature_vocab.to_compact(all_original_ids).tolist())
    assert expected_compact_rows.issubset(compact_rows_seen)
    assert all(0 <= row < feature_vocab.size for row in compact_rows_seen)

    assert len(wandb_run.logged) == 1
    metrics, logged_step = wandb_run.logged[0]
    assert logged_step == 1
    assert metrics["steering/eval_cache_hit"] == 1
    assert metrics["steering/n_completions_generated"] == 2
    assert metrics["steering/roundtrip_n_samples"] == 4
    assert (
        metrics["steering/cluster_probe_semantics"]
        == CLUSTERED_PROBE_SEMANTICS
    )
    expected_group_size = sum(
        len(group)
        for cached_target in clustered_build.payload.targets
        for group in cached_target.positive_member_groups
    ) / sum(
        len(cached_target.positive_member_groups)
        for cached_target in clustered_build.payload.targets
    )
    assert (
        metrics["steering/roundtrip_cluster_member_group_size_avg"]
        == expected_group_size
    )
    for name, value in metrics.items():
        if name == "steering/cluster_probe_semantics":
            continue
        assert isinstance(value, int | float), name
        assert math.isfinite(value), name
