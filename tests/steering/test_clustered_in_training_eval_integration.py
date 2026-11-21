"""High-level clustered in-training CORE/sample integration coverage.

This fixture crosses the production boundaries that isolated unit tests do not:
cluster metadata and effective-vocabulary validation, on-disk cache loading,
the in-training provider builder, CORE's real target-mask/context-prefix path,
sample generation's fresh-wrapper-per-prompt path, and the encoder's
original-representative-ID to compact-row remap.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch as t
import yaml
from omegaconf import OmegaConf
from safetensors.torch import save_file

import scripts.eval.base_eval as base_eval
from nanochat.core_eval import TaskMeta, TaskType
from scripts.train.conditional_encdec_train import (
    _build_cached_context_provider,
    _run_sample_generation,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.dataloader import (
    _max_selected_features_per_segment,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    canonical_dict,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.context_cache_io import (
    CacheIdentity,
    ContextCacheEntry,
    clustered_context_cache_metadata,
    context_length_cache_dir,
    save_context_cache,
)
from steering.eval.context_enumeration import (
    dedupe_context_records,
    enumerate_core_task_contexts,
    enumerate_sample_prompt_contexts,
)
from steering.eval.inference_engine import build_wrapper_for_features
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.packed_batch_io import compute_selection_hash
from steering.refined_types import PositiveInt
from steering.segmentation import token_to_char_offsets
from steering.stats_io import content_sha256
from tests.conftest import MockTokenizer, _make_gpt_config

N_SAE_FEATURES: PositiveInt = PositiveInt.parse(8)
NUM_BUCKETS: PositiveInt = PositiveInt.parse(4)
MODEL_MAX_SEQ_LEN = 128
SAMPLE_PROMPTS = ("hi", "ok")
CORE_ROWS = ({"query": "Q", "choices": ["A", "B"], "gold": 0},)
CACHED_FEATURE_IDS = (3, 7)


def test_selection_width_scan_matches_scalar_oracle_without_python_lists() -> None:
    selected_offsets = t.tensor([0, 2, 2, 7, 8], dtype=t.int64)
    negative_offsets = t.tensor([0, 1, 5, 5, 9], dtype=t.int64)
    scalar_oracle = max(
        int(selected_offsets[idx + 1] - selected_offsets[idx])
        + int(negative_offsets[idx + 1] - negative_offsets[idx])
        for idx in range(selected_offsets.numel() - 1)
    )

    assert scalar_oracle == 5
    assert (
        _max_selected_features_per_segment(selected_offsets, negative_offsets)
        == scalar_oracle
    )
    assert _max_selected_features_per_segment(t.tensor([0]), t.tensor([0])) == 0


class _RecordingProvider:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.lookups: list[tuple[int, ...]] = []

    def get(self, token_ids):
        key = tuple(int(token_id) for token_id in token_ids)
        segments = self.inner.get(key)
        if segments is not None:
            self.lookups.append(key)
        return segments


@dataclass
class _Harness:
    model: EncoderDecoderGPT
    tokenizer: MockTokenizer
    provider: _RecordingProvider
    core_base_dir: Path
    core_context_keys: set[tuple[int, ...]]
    sample_context_keys: set[tuple[int, ...]]
    compact_rows_seen: list[int]
    feature_vocab: FeatureVocab


def _write_core_bundle(base_dir: Path) -> None:
    bundle = base_dir / "eval_bundle"
    data_dir = bundle / "eval_data"
    data_dir.mkdir(parents=True)
    (data_dir / "tiny_mc.jsonl").write_text(
        "\n".join(json.dumps(row) for row in CORE_ROWS) + "\n",
        encoding="utf-8",
    )
    (bundle / "core.yaml").write_text(
        yaml.safe_dump(
            {
                "icl_tasks": [
                    {
                        "label": "tiny_mc",
                        "icl_task_type": "multiple_choice",
                        "num_fewshot": [0],
                        "continuation_delimiter": " ",
                        "dataset_uri": "tiny_mc.jsonl",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (bundle / "eval_meta_data.csv").write_text(
        "Eval Task,Random baseline\ntiny_mc,50.0\n",
        encoding="utf-8",
    )


def _build_compact_model(feature_vocab: FeatureVocab) -> EncoderDecoderGPT:
    dec_config = replace(_make_gpt_config(), sequence_len=MODEL_MAX_SEQ_LEN)
    enc_config = EncoderConfig(
        n_embd=dec_config.n_embd,
        depth=dec_config.n_layer,
        n_head=dec_config.n_head,
        n_kv_head=dec_config.n_kv_head,
        head_dim=dec_config.n_embd // dec_config.n_head,
    )
    with t.device("meta"):
        model = EncoderDecoderGPT(
            dec_config=dec_config,
            enc_config=enc_config,
            xattn_config=CrossAttentionConfig(frequency=1),
            n_sae_features=N_SAE_FEATURES,
            num_buckets=NUM_BUCKETS,
            feature_vocab=feature_vocab,
            pad_vocab_size_to=dec_config.vocab_size,
        )
    model.to_empty(device="cpu")
    t.manual_seed(123)
    model.init_weights()
    model.eval()
    return model


def _build_harness(tmp_path: Path) -> _Harness:
    tokenizer = MockTokenizer()
    mask = FeatureMask(
        FeatureMaskMode.whitelist,
        (1, 2, 3, 5, 7),
    )
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[3, 5], [2, 4]]}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(cluster_path)}},
        N_SAE_FEATURES,
    )
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, mask)
    feature_vocab = effective_feature_vocab(clusters, N_SAE_FEATURES)
    assert feature_vocab is not None
    assert feature_vocab.original_ids == (1, 2, 3, 7)

    stats_path = tmp_path / "feature_statistics.safetensors"
    save_file({"weights": t.tensor([1.0, 2.0, 3.0])}, str(stats_path), metadata={"k": "v"})
    stats_sha = content_sha256(stats_path)
    seed = 42
    feature_selection = {
        "distribution": "geometric",
        "mean": 2.0,
        "std": 2.0,
        "min_k": 1,
        "max_k": 4,
        "mode": "top_k_percentile",
        "delta_weights": "percentile_delta",
    }
    # D-seed Codex cross-review fix (Finding 1/2, HIGH): the clustered
    # prompt-cache identity is now real-seed + delta-free with
    # delta_weights RETAINED as an explicit null (not popped) --
    # steering/eval/cache_identity.py's clustered branch and
    # scripts/train/conditional_encdec_train.py's derived_hash cross-check
    # both build this exact payload shape (compute_selection_hash hashes
    # raw key presence, so popping the key would silently diverge from
    # what the production code now computes). Replaces the legacy
    # compute_canonical_feature_selection_hash call (hardcoded seed=0,
    # stats_source_sha256 folded into delta_config) this fixture used.
    prompt_selection = dict(feature_selection)
    prompt_selection["delta_weights"] = None  # ty: ignore[invalid-assignment]
    negative_selection = {
        "distribution": "geometric",
        "mean": 1.0,
        "min_k": 0,
        "max_k": 2,
    }
    selection_hash = compute_selection_hash(
        positive_selection=prompt_selection,
        negative_selection=negative_selection,
        delta_config=None,
        num_buckets=NUM_BUCKETS,
        seed=seed,
        ddp_rank=0,
        feature_mask=canonical_dict(mask),
        feature_clusters=resolved.descriptor_dict(),
        splice_batch_size=None,
    )
    identity = CacheIdentity(
        "tiny-model",
        "tiny-sae",
        0,
        "max_pool",
        selection_hash,
    )

    task_meta = TaskMeta(
        task_type=TaskType.MULTIPLE_CHOICE,
        num_fewshot=0,
        continuation_delimiter=" ",
    )
    core_records = dedupe_context_records(
        enumerate_core_task_contexts(
            tokenizer=tokenizer,
            data=list(CORE_ROWS),
            task_meta=task_meta,
            seed=1234,
            task_label="tiny_mc",
            max_per_task=1,
            max_seq_len=MODEL_MAX_SEQ_LEN,
        )
    )
    sample_records = enumerate_sample_prompt_contexts(
        prompts=SAMPLE_PROMPTS,
        tokenizer=tokenizer,
        max_seq_len=MODEL_MAX_SEQ_LEN,
    )
    records = dedupe_context_records([*core_records, *sample_records])
    entries: list[ContextCacheEntry] = []
    for record in records:
        token_ids = list(record.token_ids)
        char_end = int(token_to_char_offsets(tokenizer, token_ids)[-1])
        entries.append(
            ContextCacheEntry.from_sequences(
                token_ids,
                (
                    ContextFeatureSegment(
                        # 3 represents the live multi-member cluster [3, 5];
                        # 7 is an allowed unclustered feature.
                        feature_ids=CACHED_FEATURE_IDS,
                        buckets=(1, 3),
                        char_start=0,
                        char_end=char_end,
                    ),
                ),
            )
        )

    cache_root = tmp_path / "cache"
    cache_dir = context_length_cache_dir(
        cache_root,
        MODEL_MAX_SEQ_LEN,
        hash_leaf=selection_hash[:16],
    )
    metadata = {
        "model_max_seq_len": MODEL_MAX_SEQ_LEN,
        **clustered_context_cache_metadata(
            feature_mask=mask,
            feature_clusters=clusters,
            feature_vocab=feature_vocab,
            stats_source_sha256=stats_sha,
        ),
    }
    save_context_cache(
        cache_dir,
        entries,
        identity=identity,
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        metadata=metadata,
    )
    cfg = OmegaConf.create(
        {
            "max_seq_len": MODEL_MAX_SEQ_LEN,
            "seed": seed,
            "feature_selection": feature_selection,
            "negative_selection": negative_selection,
            "conditioning": {
                "context_cache_dir": str(cache_root),
                "n_sae_features": N_SAE_FEATURES,
                "num_buckets": NUM_BUCKETS,
                "identity": identity.as_dict(),
            },
        }
    )
    provider = _build_cached_context_provider(
        cfg,
        stats_path=stats_path,
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        feature_mask=mask,
        feature_clusters=clusters,
        feature_vocab=feature_vocab,
    )
    assert provider is not None
    recording_provider = _RecordingProvider(provider)

    model = _build_compact_model(feature_vocab)
    compact_rows_seen: list[int] = []

    def _capture_compact_rows(_module, args) -> None:
        compact_rows_seen.extend(int(row) for row in args[0].detach().flatten())

    model.encoder.feature_embed.register_forward_pre_hook(_capture_compact_rows)
    core_base_dir = tmp_path / "nanochat"
    _write_core_bundle(core_base_dir)
    return _Harness(
        model=model,
        tokenizer=tokenizer,
        provider=recording_provider,
        core_base_dir=core_base_dir,
        core_context_keys={tuple(record.token_ids) for record in core_records},
        sample_context_keys={tuple(record.token_ids) for record in sample_records},
        compact_rows_seen=compact_rows_seen,
        feature_vocab=feature_vocab,
    )


def _assert_original_ids_reached_compact_rows(harness: _Harness) -> None:
    feature_vocab = harness.feature_vocab
    expected = feature_vocab.to_compact(
        t.tensor(CACHED_FEATURE_IDS, dtype=t.int64)
    ).tolist()
    assert expected == [2, 3]
    assert set(expected).issubset(harness.compact_rows_seen)
    assert all(0 <= row < feature_vocab.size for row in harness.compact_rows_seen)


def test_real_clustered_cache_drives_in_training_core_through_compact_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _build_harness(tmp_path)
    monkeypatch.setattr(base_eval, "get_base_dir", lambda: str(harness.core_base_dir))
    monkeypatch.setattr(base_eval, "download_file_with_lock", lambda *args, **kwargs: None)
    wrapper = build_wrapper_for_features(
        harness.model,
        harness.tokenizer,
        features=None,
        context_provider=harness.provider,
    )

    results = base_eval.evaluate_core(
        wrapper,
        harness.tokenizer,
        "cpu",
        max_per_task=1,
    )

    assert set(results["results"]) == {"tiny_mc"}
    assert math.isfinite(results["core_metric"])
    assert harness.core_context_keys.issubset(set(harness.provider.lookups))
    cached = harness.provider.inner.get(next(iter(harness.core_context_keys)))
    assert cached is not None
    assert cached[0].feature_ids == CACHED_FEATURE_IDS
    _assert_original_ids_reached_compact_rows(harness)


def test_real_clustered_cache_drives_in_training_sample_helper_for_every_prompt(
    tmp_path: Path,
) -> None:
    harness = _build_harness(tmp_path)

    _run_sample_generation(
        orig_model=harness.model,
        tokenizer=harness.tokenizer,
        device=t.device("cpu"),
        sample_prompts=SAMPLE_PROMPTS,
        cached_context_provider=harness.provider,
    )

    assert harness.sample_context_keys.issubset(set(harness.provider.lookups))
    _assert_original_ids_reached_compact_rows(harness)
