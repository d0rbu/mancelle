"""Real-scale issue #714 integration gates.

These tests are intentionally excluded from the default suite. They use the
canonical 65,536-feature Gemma-3 artifacts and the exact queued-run model
builder, and therefore must never be described as ordinary CI smoke tests.

The model-only GPU test allocates the queued encoder-decoder architecture with
the production audited whitelist on a 24 GiB-class device, backpropagates through
its full 12-layer feature encoder, and differentially hashes full-decoder eval
logits plus every training gradient for an active cluster file that becomes
identity after masking. The combined gate feeds an exact complete-segment prefix
of a real production row through both independent preselection/loader arms and
the 499,392,150-parameter whitelist-compacted model, including its compact FP
head and a production optimizer step. The exact 2,074-row production cluster
artifact is pinned separately by ``test_issue768_production_artifacts.py``.
The combined gate uses the issue-#831 context-1024 production packed family.
The artifact test separately loads and processes one complete 16,384-row,
context-2048 production packed shard (roughly 1.5 GB compressed), including a
serialized byte-for-byte identity comparison. All skip loudly when required
local artifacts or capacity are unavailable.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch as t
import torch.nn.functional as F

from errors import FeatureVocabError
from scripts.preprocess import precompute_selected_features as selection_producer
from scripts.preprocess.precompute_selected_features import (
    _materialize_splice_segment_selection,
)
from scripts.train.conditional_encdec_train import (
    build_encdec_meta_model_from_training_cfg,
)
from steering.conditional.delta import (
    DeltaMode,
    DeltaSamplingWeights,
    FeatureDeltaConfig,
    compute_doc_clustered_counterfactual_deltas,
    compute_doc_counterfactual_deltas,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    FeatureClusterConsolidation,
    build_clustered_selection_shard_integrity,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import canonical_dict, resolve_feature_mask
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.conditional.thresholding import ThresholdType
from steering.overlay_io import (
    OverlayKind,
    glob_overlay_shards,
    overlay_shard_path,
    write_manifest,
)
from steering.packed_batch_io import (
    PackedRowShard,
    load_packed_row_shard,
    load_packed_row_shard_bytes,
    load_selection_shard,
    load_selection_shard_bytes,
    save_packed_row_shard,
)

pytestmark = [
    pytest.mark.very_slow,
    pytest.mark.requires_feature_data,
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_ROOT = Path("/home/d0rb/Documents/Github/nano-gpt-pretrain-steer")
_ASSET_ROOT = Path(os.environ.get("ISSUE714_ASSET_ROOT", str(_CANONICAL_ROOT)))
_REAL_FEATURE_DIR = _ASSET_ROOT / (
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
    "m_gemma-3-4b-pt/base/context_independent/no_bos"
)
_REAL_WHITELIST = _ASSET_ROOT / (
    "out/autointerp_filter/"
    "gemma3_4bpt_l22_f65k_ci_nobos_freqmask_n40x40_root_20260710_145543/"
    "manual_feature_filter_with_descriptions.yaml"
)
_REAL_PACKED_DIR = Path(
    os.environ.get(
        "ISSUE714_PACKED_DIR",
        str(
            _REAL_FEATURE_DIR / "splice/cut_frac0.15/seed42/pct/623c372d3f121b18/"
            "tok/dc6308e6116cbfe0/c2048/sall"
        ),
    )
)
_REAL_C1024_PACKED_DIR = Path(
    os.environ.get(
        "ISSUE831_PACKED_DIR",
        str(
            _REAL_FEATURE_DIR / "splice/cut_frac0.15/seed42/pct/623c372d3f121b18/"
            "tok/dc6308e6116cbfe0/c1024/sall"
        ),
    )
)
_TRAINING_CONFIG_DIR = _REPO_ROOT / "configs/training"
_QUEUED_CONFIG = "cluster_fp/conditional_encdec_splice_cf_gemma3pt_8xpro6000"
_N_SAE = 65_536
_TOKEN_VOCAB = 32_768
_GIB = 1024**3
# ``feature_percentile_segment_topk`` encodes ``k`` in the integer part and a
# required two-decimal percentile floor in the fractional part. 8.01 is the
# least restrictive valid top-8 policy and excludes only zero-percentile
# packed entries; 8.0 is invalid rather than a top-k-only shorthand.
_PACKED_TOP8_THRESHOLD = 8.01


def _require(path: Path, purpose: str) -> Path:
    if not path.exists():
        pytest.skip(f"missing real {purpose}: {path}")
    return path


def _available_ram_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return values["MemAvailable"] / _GIB


def _real_mask():
    mask = resolve_feature_mask(
        {
            "feature_mask": {
                "mode": "whitelist",
                "path": str(_require(_REAL_WHITELIST, "audited manual whitelist")),
            }
        },
        _N_SAE,
    )
    assert mask is not None
    assert len(mask.ids) == 16_254
    return mask


def _disallowed_ids(mask) -> t.Tensor:
    return t.nonzero(
        ~mask.allowed_universe_mask(_N_SAE),
        as_tuple=False,
    ).squeeze(1)


def _write_cluster_state(
    tmp_path: Path,
    groups: list[list[int]],
    *,
    mask,
    name: str = "clusters.json",
) -> FeatureClusterConsolidation:
    source = tmp_path / name
    source.write_text(
        json.dumps({"schema_version": 1, "clusters": groups}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(source)}},
        _N_SAE,
    )
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _cuda_cleanup() -> None:
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
        t.cuda.synchronize()


def _sha256_named_tensors(
    tensors,
) -> str:
    """Hash names, dtype/shape metadata, and exact raw tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if tensor is None:
            digest.update(b"<none>\0")
            continue
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        raw = tensor.detach().contiguous().view(t.uint8).cpu().numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _sha256_nested(value: Any) -> str:
    """Hash a nested optimizer state without retaining a second full copy."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        digest.update(type(item).__qualname__.encode("utf-8"))
        digest.update(b"\0")
        if isinstance(item, t.Tensor):
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(repr(tuple(item.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(item.detach().contiguous().view(t.uint8).cpu().numpy().tobytes())
            return
        if isinstance(item, dict):
            digest.update(str(len(item)).encode("ascii"))
            digest.update(b"\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode("ascii"))
            digest.update(b"\0")
            for child in item:
                update(child)
            return
        digest.update(repr(item).encode("utf-8"))
        digest.update(b"\0")

    update(value)
    return digest.hexdigest()


def _extract_complete_real_prefix(
    packed: PackedRowShard,
    *,
    context_len: int,
) -> tuple[PackedRowShard, int]:
    """Extract a real row prefix ending on an existing segment boundary."""
    chosen_row = None
    chosen_n_segments = None
    for row_idx in range(packed.n_rows):
        seg_start = int(packed.row_seg_offsets[row_idx])
        seg_end = int(packed.row_seg_offsets[row_idx + 1])
        matching = (packed.seg_token_end[seg_start:seg_end].to(t.int64) == context_len).nonzero(
            as_tuple=False
        )
        if matching.numel() > 0:
            chosen_row = row_idx
            chosen_n_segments = int(matching[0, 0]) + 1
            break

    assert chosen_row is not None, (
        f"real packed shard has no row with a complete segment ending at {context_len}"
    )
    assert chosen_n_segments is not None
    source_seg_start = int(packed.row_seg_offsets[chosen_row])
    source_seg_end = source_seg_start + chosen_n_segments
    source_feature_start = int(packed.seg_feat_offsets[source_seg_start])
    source_feature_end = int(packed.seg_feat_offsets[source_seg_end])

    prefix = PackedRowShard(
        tokens=packed.tokens[chosen_row : chosen_row + 1, : context_len + 1].contiguous().clone(),
        row_seg_offsets=t.tensor(
            [0, chosen_n_segments],
            dtype=packed.row_seg_offsets.dtype,
        ),
        seg_token_start=packed.seg_token_start[source_seg_start:source_seg_end].clone(),
        seg_token_end=packed.seg_token_end[source_seg_start:source_seg_end].clone(),
        seg_doc_id=packed.seg_doc_id[source_seg_start:source_seg_end].clone(),
        seg_feat_offsets=(
            packed.seg_feat_offsets[source_seg_start : source_seg_end + 1] - source_feature_start
        ).clone(),
        feature_ids=packed.feature_ids[source_feature_start:source_feature_end].clone(),
        feature_pcts=packed.feature_pcts[source_feature_start:source_feature_end].clone(),
    )
    assert prefix.tokens.shape == (1, context_len + 1)
    assert prefix.total_segments == chosen_n_segments
    assert int(prefix.seg_token_start[0]) == 0
    assert int(prefix.seg_token_end[-1]) == context_len
    assert t.equal(prefix.seg_token_start[1:], prefix.seg_token_end[:-1])
    assert prefix.total_features == source_feature_end - source_feature_start
    return prefix, chosen_row


@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
def test_full_queued_gpu_model_uses_mask_compaction_and_preserves_identity_parity(
    hydra_compose,
    tmp_path: Path,
) -> None:
    """Backprop through the exact production-sized model and cluster mapping.

    The audited whitelist contracts 65,536 SAE rows to 16,254. The two
    active post-mask clusters then remove two consolidated-away members, so the
    test vocabulary carries exactly 16,252 rows. Away ids are physically
    absent and fail the original-to-compact remap; representatives remain legal.

    The same full model then exercises the equivalence boundary requested in
    review: an *active* cluster file whose every group has one post-mask
    survivor. Full-sequence eval logits/loss, training loss, and the raw bytes
    of every parameter gradient must match the unclustered input exactly.
    """
    _cuda_cleanup()
    free_bytes, total_bytes = t.cuda.mem_get_info()
    if total_bytes < 23 * _GIB or free_bytes < 20 * _GIB:
        pytest.skip(
            "full queued model gate needs a 24 GiB-class GPU with at least "
            f"20 GiB free; total={total_bytes / _GIB:.1f}, "
            f"free={free_bytes / _GIB:.1f} GiB"
        )

    mask = _real_mask()
    allowed = t.nonzero(mask.allowed_universe_mask(_N_SAE), as_tuple=False).squeeze(1)
    assert allowed.numel() == 16_254
    a, b, c, d, unclustered, reference_only = [int(x) for x in allowed[100:106].tolist()]
    masked_member = int(_disallowed_ids(mask)[0])
    clusters = _write_cluster_state(
        tmp_path,
        [[masked_member, a, b], [c, d]],
        mask=mask,
    )
    representatives, reduced_pcts = clusters.consolidate_segment(
        t.tensor([b, d, unclustered], dtype=t.int64),
        t.tensor([0.95, 0.80, 0.70], dtype=t.float32),
    )
    assert representatives.tolist() == sorted([a, c, unclustered])
    assert sorted(reduced_pcts.tolist()) == pytest.approx([0.70, 0.80, 0.95])

    # Exercise the corrected production splice order at full SAE width: mask
    # and independently max-consolidate current/reference sides, then subtract
    # in representative space before materialization.
    current_ids = t.tensor([b, d, unclustered], dtype=t.int64)
    current_pcts = t.tensor([0.95, 0.80, 0.10], dtype=t.float32)
    reference_ids = t.tensor([unclustered, reference_only], dtype=t.int64)
    reference_pcts = t.tensor([0.90, 0.70], dtype=t.float32)
    delta_result = compute_doc_clustered_counterfactual_deltas(
        [current_ids],
        [current_pcts.clone()],
        [current_pcts],
        [(reference_ids, reference_pcts.clone(), reference_pcts)],
        clusters,
    )[0]
    assert delta_result is not None
    percentile_table = t.zeros((_N_SAE, 99), dtype=t.float32)
    splice_pipeline = FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.normal,
                mean=2.0,
                std=0.01,
                min_k=2,
                max_k=3,
                mode=SamplingMode.top_k_percentile,
                delta_weights=DeltaSamplingWeights.percentile_delta,
            ),
            percentile_table=percentile_table,
            rng=t.Generator().manual_seed(714),
        ),
        boundaries=t.zeros((_N_SAE, 1), dtype=t.float64),
        negative_bundle=NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.normal,
                mean=2.0,
                std=0.01,
                min_k=2,
                max_k=3,
            ),
            n_sae_features=_N_SAE,
            rng=t.Generator().manual_seed(715),
            feature_mask=mask,
        ),
        effective_buckets=1,
        delta_bundle=DeltaBundle(
            config=FeatureDeltaConfig(
                mode=DeltaMode.splice_counterfactual,
                neg_delta_ratio=1.0,
            ),
            percentile_table=percentile_table,
        ),
        feature_mask=mask,
    )
    selected, buckets, selected_pcts, negatives = _materialize_splice_segment_selection(
        raw_fids=current_ids,
        raw_mags=current_pcts,
        raw_pcts=current_pcts,
        delta_result=delta_result,
        pipeline_config=splice_pipeline,
        feature_clusters=clusters,
    )
    assert selected == [a, c]
    assert buckets == [0, 0]
    assert selected_pcts == [round(0.95 * 255), round(0.80 * 255)]
    assert set(negatives) == {unclustered, reference_only}
    del splice_pipeline, percentile_table, delta_result

    vocab = effective_feature_vocab(clusters, _N_SAE)
    assert vocab is not None and vocab.size == 16_252
    away = t.tensor([b, d], dtype=t.int64)
    with pytest.raises(FeatureVocabError, match="outside the encoder vocabulary"):
        vocab.to_compact(away)
    representative_compact = vocab.to_compact(t.tensor([a, c], dtype=t.int64))
    assert bool((representative_compact >= 0).all())

    cfg = hydra_compose(str(_TRAINING_CONFIG_DIR), _QUEUED_CONFIG, None)
    model = None
    encoded = None
    projection = None
    loss = None
    try:
        with t.device("meta"):
            model = build_encdec_meta_model_from_training_cfg(
                cfg,
                base_vocab=_TOKEN_VOCAB,
                n_sae_features=_N_SAE,
                feature_vocab=vocab,
            )
        assert model.dec_config.n_layer == 12
        assert model.encoder.config.n_layer == 12
        assert len(model.cross_attn_layers) == 12
        assert len(model.encoder.value_embeds) == 6
        assert model.encoder.feature_embed.weight.shape[0] == 16_252

        model.to_empty(device="cuda")
        # Production materializes and initializes shared state, then reseeds
        # and initializes once more for a fresh per-config run.
        t.manual_seed(714)
        model.init_weights()
        t.manual_seed(714 * 1_000_003)
        model.init_weights()
        model.train()

        feature_ids = representatives.reshape(1, 1, -1).to("cuda")
        magnitudes = t.tensor([[[0, 1, 0]]], dtype=t.int64, device="cuda")
        mask_tensor = t.ones_like(feature_ids, dtype=t.bool)
        encoded = model.encoder(feature_ids, magnitudes, mask_tensor)
        projection = t.randn(
            encoded.values.shape,
            dtype=encoded.values.dtype,
            device=encoded.values.device,
            generator=t.Generator(device="cuda").manual_seed(714),
        )
        loss = (
            encoded.values[:, encoded.layout.valid_mask]
            * projection[:, encoded.layout.valid_mask]
        ).sum()
        loss.backward()
        t.cuda.synchronize()

        feature_grad = model.encoder.feature_embed.weight.grad
        assert feature_grad is not None
        rep_compact = vocab.to_compact(representatives).to("cuda")
        assert bool((feature_grad[rep_compact].abs().sum(dim=1) > 0).all())
        for layer, value_embed in model.encoder.value_embeds.items():
            value_grad = value_embed.weight.grad
            assert isinstance(value_grad, t.Tensor), f"value_embeds[{layer}] got no gradient"
        del loss, projection, encoded
        loss = projection = encoded = None

        # Active descriptor, numerically identity after the already-resolved
        # mask: each pair contains one masked ID and one allowed ID.
        identity_original_ids = allowed[300:324].to(t.int64)
        disallowed = _disallowed_ids(mask)
        identity_groups = [
            sorted([int(disallowed[index]), int(identity_original_ids[index])])
            for index in range(identity_original_ids.numel())
        ]
        identity_clusters = _write_cluster_state(
            tmp_path,
            identity_groups,
            mask=mask,
            name="identity_after_real_mask.json",
        )
        assert not bool(identity_clusters.consolidated_away.any())
        assert t.equal(
            identity_clusters.rep_of,
            t.arange(_N_SAE, dtype=t.int64),
        )
        assert identity_clusters.effective_negative_mask() == mask
        identity_pcts = t.linspace(
            0.20,
            0.99,
            identity_original_ids.numel(),
            dtype=t.float32,
        )
        clustered_ids, clustered_pcts = identity_clusters.consolidate_segment(
            identity_original_ids,
            identity_pcts,
        )
        assert t.equal(clustered_ids, identity_original_ids)
        assert t.equal(clustered_pcts, identity_pcts)

        sequence_len = int(cfg.max_seq_len)
        assert sequence_len == 1_024
        n_segments = sequence_len // 128
        unclustered_feature_ids = (
            identity_original_ids.reshape(1, 1, -1)
            .expand(1, n_segments, -1)
            .contiguous()
            .to("cuda")
        )
        clustered_feature_ids = (
            clustered_ids.reshape(1, 1, -1).expand(1, n_segments, -1).contiguous().to("cuda")
        )
        assert t.equal(clustered_feature_ids, unclustered_feature_ids)
        compact_ids = vocab.to_compact(identity_original_ids)
        assert bool((compact_ids >= 0).all())
        assert not t.equal(compact_ids, identity_original_ids)

        magnitudes = (
            t.arange(identity_original_ids.numel(), dtype=t.int64)
            .remainder(int(cfg.num_buckets))
            .reshape(1, 1, -1)
            .expand(1, n_segments, -1)
            .contiguous()
            .to("cuda")
        )
        active_mask = t.ones_like(unclustered_feature_ids, dtype=t.bool)
        seg_map = (
            t.arange(sequence_len, dtype=t.int64, device="cuda")
            .div(128, rounding_mode="floor")
            .reshape(1, -1)
        )
        token_generator = t.Generator(device="cuda").manual_seed(714)
        inputs = t.randint(
            0,
            _TOKEN_VOCAB,
            (1, sequence_len),
            generator=token_generator,
            device="cuda",
        )
        targets = t.randint(
            0,
            _TOKEN_VOCAB,
            (1, sequence_len),
            generator=token_generator,
            device="cuda",
        )
        fp_targets = t.zeros(
            (1, n_segments, vocab.size),
            dtype=t.float32,
            device="cuda",
        )
        seg_doc_id = t.zeros((1, n_segments), dtype=t.int64, device="cuda")

        model.zero_grad(set_to_none=True)
        model.eval()
        with t.no_grad():
            unclustered_logits = model(
                inputs,
                None,
                unclustered_feature_ids,
                magnitudes,
                active_mask,
                seg_map,
            )
            unclustered_eval_loss = F.cross_entropy(
                unclustered_logits.reshape(-1, unclustered_logits.shape[-1]),
                targets.reshape(-1),
            )
            unclustered_logits_sha = _sha256_named_tensors([("eval_logits", unclustered_logits)])
            del unclustered_logits
            clustered_logits = model(
                inputs,
                None,
                clustered_feature_ids,
                magnitudes,
                active_mask,
                seg_map,
            )
            clustered_eval_loss = F.cross_entropy(
                clustered_logits.reshape(-1, clustered_logits.shape[-1]),
                targets.reshape(-1),
            )
            clustered_logits_sha = _sha256_named_tensors([("eval_logits", clustered_logits)])
            del clustered_logits
        assert t.equal(clustered_eval_loss, unclustered_eval_loss)
        assert clustered_logits_sha == unclustered_logits_sha

        model.train()
        model.zero_grad(set_to_none=True)
        unclustered_training_result = model(
            inputs,
            targets,
            unclustered_feature_ids,
            magnitudes,
            active_mask,
            seg_map,
            fp_targets=fp_targets,
            seg_doc_id=seg_doc_id,
        )
        unclustered_training_loss = unclustered_training_result.loss
        unclustered_training_loss.backward()
        unclustered_grad_sha = _sha256_named_tensors(
            (name, parameter.grad) for name, parameter in model.named_parameters()
        )
        model.zero_grad(set_to_none=True)
        clustered_training_result = model(
            inputs,
            targets,
            clustered_feature_ids,
            magnitudes,
            active_mask,
            seg_map,
            fp_targets=fp_targets,
            seg_doc_id=seg_doc_id,
        )
        clustered_training_loss = clustered_training_result.loss
        clustered_training_loss.backward()
        clustered_grad_sha = _sha256_named_tensors(
            (name, parameter.grad) for name, parameter in model.named_parameters()
        )
        assert t.equal(
            clustered_training_loss.detach(),
            unclustered_training_loss.detach(),
        )
        assert clustered_grad_sha == unclustered_grad_sha
        assert unclustered_grad_sha != hashlib.sha256().hexdigest()
        assert model.feature_pred_head is not None
        assert model.feature_pred_head.weight.grad is not None
        assert bool(model.feature_pred_head.weight.grad.abs().sum() > 0)
    finally:
        del loss
        del projection
        del encoded
        del model
        _cuda_cleanup()


@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
def test_real_row_reaches_full_model_with_exact_identity_cluster_parity(
    hydra_compose,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Join the real-data and full-model parity proofs in one differential.

    The queued 65k run has no trained checkpoint yet. This gate therefore uses
    its exact production builder and initialization, but unlike the neighboring
    model-only gate, every model input comes from the production precomputed
    loader over a serialized prefix of a real packed row. The prefix ends at an
    existing 1,024-token segment boundary; no segment or feature vector is
    synthesized, truncated, or recomputed.

    Both arms run the real selection producer and loader independently. The
    active cluster file pairs every selected real feature with a feature excluded
    by the audited whitelist, making it identity only *after* filtering.
    Selection bytes, all loader tensors, eval logits/loss, training loss, every
    gradient, a production MuonAdamW step, all updated parameters, optimizer
    state, and RNG continuation must be exact.
    """
    _cuda_cleanup()
    free_bytes, total_bytes = t.cuda.mem_get_info()
    if total_bytes < 23 * _GIB or free_bytes < 20 * _GIB:
        pytest.skip(
            "combined real-row/full-model gate needs a 24 GiB-class GPU with "
            f"at least 20 GiB free; total={total_bytes / _GIB:.1f}, "
            f"free={free_bytes / _GIB:.1f} GiB"
        )
    # build_shared_state applies this production prerequisite before optimizer
    # construction. The full model has more than eight distinct Muon matrix
    # shapes, so pytest's process default would fail before completing a step.
    monkeypatch.setattr(t._dynamo.config, "cache_size_limit", 64)

    packed_paths = glob_overlay_shards(
        _require(_REAL_C1024_PACKED_DIR, "issue-#831 context-1024 packed-row directory")
    )
    full_shards = [path for path in packed_paths if path.stat().st_size > 700_000_000]
    if not full_shards:
        pytest.skip("no complete ~773 MB issue-#831 production packed shard is present")
    source_path = full_shards[0]
    source_compressed = source_path.read_bytes()
    source = load_packed_row_shard_bytes(source_compressed)
    del source_compressed
    assert source.tokens.shape == (16_384, 1_025)
    assert source.total_segments > 100_000
    real_prefix, source_row = _extract_complete_real_prefix(
        source,
        context_len=1_024,
    )
    assert 0 <= source_row < source.n_rows
    assert real_prefix.total_segments >= 8
    assert real_prefix.total_features > 10_000
    del source
    gc.collect()

    packed_dir = tmp_path / "real_prefix_packed"
    packed_dir.mkdir()
    shard_stem = source_path.stem.split(".")[0]
    packed_path = overlay_shard_path(packed_dir, shard_stem)
    save_packed_row_shard(real_prefix, packed_path)
    segment_lengths = real_prefix.seg_token_end.to(t.int64) - real_prefix.seg_token_start.to(
        t.int64
    )
    write_manifest(
        packed_dir,
        "issue714-real-prefix-packed",
        OverlayKind.PACKED_ROW,
        metadata={
            "type": "packed_rows",
            "context_len": real_prefix.context_len,
            "max_n_seg": real_prefix.total_segments,
            "max_seg_size": int(segment_lengths.max()),
            "source_shard": source_path.name,
            "source_row": source_row,
        },
    )

    mask = _real_mask()
    vocab = FeatureVocab.from_mask(mask, _N_SAE)
    assert vocab is not None and vocab.size == 16_254

    def real_pipeline() -> FeaturePipelineConfig:
        return FeaturePipelineConfig(
            selection_config=ThresholdSelectionConfig(
                selection=ThresholdSelection(
                    ThresholdType.feature_percentile_segment_topk,
                    _PACKED_TOP8_THRESHOLD,
                ),
                precomputed_thresholds=t.zeros(_N_SAE, dtype=t.float32),
            ),
            boundaries=t.zeros((_N_SAE, 1), dtype=t.float64),
            negative_bundle=None,
            effective_buckets=2,
            feature_mask=mask,
        )

    unclustered_dir = tmp_path / "real_prefix_unclustered"
    clustered_dir = tmp_path / "real_prefix_clustered"
    unclustered_dir.mkdir()
    clustered_dir.mkdir()
    selection_producer.select_features_for_shard(
        packed_path,
        real_pipeline(),
        unclustered_dir,
        shard_stem,
    )
    unclustered_path = overlay_shard_path(unclustered_dir, shard_stem)
    unclustered_selection = load_selection_shard(unclustered_path)
    selected_real_ids = sorted(set(unclustered_selection.sel_fids.to(t.int64).tolist()))
    assert selected_real_ids
    assert len(selected_real_ids) <= real_prefix.total_segments * 8
    assert len(selected_real_ids) <= len(mask.ids)
    mask.assert_no_masked(t.tensor(selected_real_ids, dtype=t.int64))

    disallowed = _disallowed_ids(mask)
    identity_groups = [
        sorted([int(disallowed[index]), feature_id])
        for index, feature_id in enumerate(selected_real_ids)
    ]
    identity_clusters = _write_cluster_state(
        tmp_path,
        identity_groups,
        mask=mask,
        name="real_row_identity_after_mask.json",
    )
    assert identity_clusters.is_identity_after_mask is True
    assert not bool(identity_clusters.consolidated_away.any())
    assert t.equal(
        identity_clusters.rep_of,
        t.arange(_N_SAE, dtype=t.int64),
    )

    selection_producer.select_features_for_shard(
        packed_path,
        real_pipeline(),
        clustered_dir,
        shard_stem,
        feature_clusters=identity_clusters,
    )
    clustered_path = overlay_shard_path(clustered_dir, shard_stem)
    assert clustered_path.read_bytes() == unclustered_path.read_bytes()
    clustered_selection = load_selection_shard_bytes(clustered_path.read_bytes())
    for name in (
        "sel_offsets",
        "sel_fids",
        "sel_buckets",
        "sel_pcts",
        "neg_offsets",
        "neg_fids",
    ):
        clustered_value = getattr(clustered_selection, name)
        unclustered_value = getattr(unclustered_selection, name)
        assert clustered_value.dtype == unclustered_value.dtype, name
        assert t.equal(clustered_value, unclustered_value), name

    packed_fingerprint = packed_row_fingerprint(packed_dir)
    base_metadata = {
        "packed_rows": {"packed_row_fingerprint": packed_fingerprint},
        "feature_mask": canonical_dict(mask),
    }
    write_manifest(
        unclustered_dir,
        "issue714-real-prefix-unclustered",
        OverlayKind.PACKED_ROW,
        metadata=base_metadata,
        base_shard_paths=[packed_path],
    )
    cluster_metadata = identity_clusters.metadata_dict()
    cluster_metadata["status"] = "complete"
    clustered_metadata = {
        **base_metadata,
        "feature_clusters": cluster_metadata,
        SELECTION_SHARD_INTEGRITY_KEY: build_clustered_selection_shard_integrity([clustered_path]),
    }
    write_manifest(
        clustered_dir,
        "issue714-real-prefix-clustered",
        OverlayKind.PACKED_ROW,
        metadata=clustered_metadata,
        base_shard_paths=[packed_path],
    )

    def next_real_batch(
        selection_dir: Path,
        clusters: FeatureClusterConsolidation | None,
    ):
        return next(
            precomputed_data_loader(
                packed_row_dir=packed_dir,
                B=1,
                feature_pipeline=real_pipeline(),
                device="cuda",
                selection_dir=selection_dir,
                effective_buckets=2,
                n_sae_features=_N_SAE,
                feature_clusters=clusters,
            )
        )

    unclustered_batch = next_real_batch(unclustered_dir, None)
    clustered_batch = next_real_batch(clustered_dir, identity_clusters)
    for index, (clustered_value, unclustered_value) in enumerate(
        zip(clustered_batch[:-1], unclustered_batch[:-1], strict=True)
    ):
        if clustered_value is None or unclustered_value is None:
            assert clustered_value is unclustered_value, f"batch[{index}]"
        else:
            assert clustered_value.dtype == unclustered_value.dtype, f"batch[{index}]"
            assert t.equal(clustered_value, unclustered_value), f"batch[{index}]"
    clustered_state = dict(clustered_batch[-1])
    assert clustered_state.pop("feature_clusters_descriptor") == (
        identity_clusters.resolved.descriptor_dict()
    )
    assert "feature_clusters_descriptor" not in unclustered_batch[-1]
    assert clustered_state == unclustered_batch[-1]

    assert unclustered_batch[0].shape == (1, 1_024)
    assert t.equal(
        unclustered_batch[0].cpu(),
        real_prefix.tokens[:, :-1].to(t.int64),
    )
    assert t.equal(
        unclustered_batch[1].cpu(),
        real_prefix.tokens[:, 1:].to(t.int64),
    )
    active_original_ids = clustered_batch[2][clustered_batch[4]].cpu()
    assert set(active_original_ids.tolist()) == set(selected_real_ids)
    compact_ids = vocab.to_compact(active_original_ids)
    assert bool((compact_ids >= 0).all())
    assert bool((compact_ids != active_original_ids).any())

    cfg = hydra_compose(str(_TRAINING_CONFIG_DIR), _QUEUED_CONFIG, None)
    assert int(cfg.max_seq_len) == real_prefix.context_len

    def run_full_model_arm(batch) -> dict[str, Any]:
        model = None
        optimizer = None
        try:
            with t.device("meta"):
                model = build_encdec_meta_model_from_training_cfg(
                    cfg,
                    base_vocab=_TOKEN_VOCAB,
                    n_sae_features=_N_SAE,
                    feature_vocab=vocab,
                )
            model.to_empty(device="cuda")
            t.manual_seed(714)
            model.init_weights()
            t.manual_seed(714 * 1_000_003)
            model.init_weights()
            model.feature_mask = mask
            n_parameters = sum(parameter.numel() for parameter in model.parameters())
            assert n_parameters == 499_392_150
            assert model.encoder.feature_embed.weight.shape[0] == vocab.size
            assert model.feature_prediction_width == vocab.size
            initial_model_sha = _sha256_named_tensors(model.named_parameters())

            optimizer = model.setup_optimizer(
                unembedding_lr=float(cfg.unembedding_lr),
                embedding_lr=float(cfg.embedding_lr),
                scalar_lr=float(cfg.scalar_lr),
                matrix_lr=float(cfg.matrix_lr),
                weight_decay=float(cfg.weight_decay),
                adam_betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
                encoder_embedding_lr=float(cfg.encoder_embedding_lr),
                encoder_matrix_lr=float(cfg.encoder_matrix_lr),
                xattn_matrix_lr=float(cfg.xattn_matrix_lr),
            )

            model.eval()
            with t.no_grad():
                logits = model(
                    batch[0],
                    None,
                    batch[2],
                    batch[3],
                    batch[4],
                    batch[6],
                )
                eval_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch[1].reshape(-1),
                )
                logits_sha = _sha256_named_tensors([("eval_logits", logits)])
            del logits

            model.train()
            optimizer.zero_grad(set_to_none=True)
            full_fp_targets = batch[8]
            assert full_fp_targets is not None
            compact_original_ids = t.tensor(
                vocab.original_ids,
                dtype=t.int64,
                device=full_fp_targets.device,
            )
            compact_fp_targets = full_fp_targets.index_select(
                2,
                compact_original_ids,
            )
            training_result = model(
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
                batch[6],
                fp_targets=compact_fp_targets,
                seg_doc_id=batch[7],
            )
            training_loss = training_result.loss
            training_loss.backward()
            t.cuda.synchronize()
            assert model.feature_pred_head is not None
            assert model.feature_pred_head.weight.grad is not None
            assert bool(model.feature_pred_head.weight.grad.abs().sum() > 0)
            grad_sha = _sha256_named_tensors(
                (name, parameter.grad) for name, parameter in model.named_parameters()
            )
            assert grad_sha != hashlib.sha256().hexdigest()
            optimizer.step()
            t.cuda.synchronize()
            updated_model_sha = _sha256_named_tensors(model.named_parameters())
            optimizer_sha = _sha256_nested(optimizer.state_dict())
            rng_sha = _sha256_named_tensors([("cuda_rng_state", t.cuda.get_rng_state())])
            return {
                "n_parameters": n_parameters,
                "initial_model_sha": initial_model_sha,
                "eval_loss": eval_loss.detach().cpu().clone(),
                "logits_sha": logits_sha,
                "training_loss": training_loss.detach().cpu().clone(),
                "grad_sha": grad_sha,
                "updated_model_sha": updated_model_sha,
                "optimizer_sha": optimizer_sha,
                "rng_sha": rng_sha,
            }
        finally:
            del optimizer
            del model
            _cuda_cleanup()

    unclustered_result = run_full_model_arm(unclustered_batch)
    clustered_result = run_full_model_arm(clustered_batch)
    assert clustered_result.keys() == unclustered_result.keys()
    for name in clustered_result:
        clustered_value = clustered_result[name]
        unclustered_value = unclustered_result[name]
        if isinstance(clustered_value, t.Tensor):
            assert clustered_value.dtype == unclustered_value.dtype, name
            assert t.equal(clustered_value, unclustered_value), name
        else:
            assert clustered_value == unclustered_value, name


def test_complete_real_packed_shard_preselection_is_representative_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run production consolidation across one full real packed shard.

    This is intentionally a full-shard gate, not a handful of copied rows. A
    RAM guard avoids destabilizing shared machines while another production
    packer is resident; rerun when at least 24 GiB is available.
    """
    available_gib = _available_ram_gib()
    if available_gib < 24:
        pytest.skip(
            f"full real packed-shard gate needs >=24 GiB available RAM; "
            f"have {available_gib:.1f} GiB"
        )
    packed_paths = glob_overlay_shards(
        _require(_REAL_PACKED_DIR, "context-2048 packed-row directory")
    )
    if not packed_paths:
        pytest.skip(f"no real packed shards under {_REAL_PACKED_DIR}")
    full_shards = [path for path in packed_paths if path.stat().st_size > 1 * _GIB]
    if not full_shards:
        pytest.skip("no complete ~1.5 GB production packed shard is present")
    packed_path = full_shards[0]
    packed = load_packed_row_shard(packed_path)
    assert packed.tokens.shape == (16_384, 2_049)
    assert packed.total_segments > 100_000

    offsets = packed.seg_feat_offsets.to(t.int64)
    chosen_segment = None
    chosen_ids = None
    chosen_pcts = None
    for segment_idx in range(packed.total_segments):
        start = int(offsets[segment_idx])
        end = int(offsets[segment_idx + 1])
        if end - start < 12:
            continue
        ids = packed.feature_ids[start:end].to(t.int64)
        pcts = packed.feature_pcts[start:end].to(t.float32) / 255.0
        top = t.topk(pcts, 4).indices
        selected_ids = ids[top]
        if selected_ids.unique().numel() == 4:
            chosen_segment = segment_idx
            chosen_ids = selected_ids
            chosen_pcts = pcts[top]
            break
    assert chosen_segment is not None
    assert chosen_ids is not None and chosen_pcts is not None

    first_pair = sorted(int(x) for x in chosen_ids[:2].tolist())
    second_pair = sorted(int(x) for x in chosen_ids[2:].tolist())
    clusters = _write_cluster_state(
        tmp_path,
        [first_pair, second_pair],
        mask=None,
    )
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(
                ThresholdType.feature_percentile_segment_topk,
                _PACKED_TOP8_THRESHOLD,
            ),
            precomputed_thresholds=t.zeros(_N_SAE, dtype=t.float32),
        ),
        boundaries=t.zeros(_N_SAE, 1, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=2,
        feature_mask=None,
    )
    output_dir = tmp_path / "selection"
    output_dir.mkdir()

    original_loader = selection_producer.load_packed_row_shard

    def load_once(path):
        if Path(path) == packed_path:
            return packed
        return original_loader(path)

    monkeypatch.setattr(selection_producer, "load_packed_row_shard", load_once)
    selection_producer.select_features_for_shard(
        packed_path,
        pipeline,
        output_dir,
        packed_path.stem.split(".")[0],
        feature_clusters=clusters,
    )
    selection = load_selection_shard(overlay_shard_path(output_dir, packed_path.stem.split(".")[0]))
    clusters.assert_selection_shard(
        sel_offsets=selection.sel_offsets,
        sel_fids=selection.sel_fids,
        sel_buckets=selection.sel_buckets,
        sel_pcts=selection.sel_pcts,
        neg_offsets=selection.neg_offsets,
        neg_fids=selection.neg_fids,
        expected_segments=packed.total_segments,
        effective_buckets=2,
        surface="real production packed shard",
    )
    assert int((selection.sel_offsets[1:] - selection.sel_offsets[:-1]).max()) <= 8

    away_ids = {first_pair[1], second_pair[1]}
    for away_id in away_ids:
        assert not bool((selection.sel_fids.to(t.int64) == away_id).any())
    start = int(selection.sel_offsets[chosen_segment])
    end = int(selection.sel_offsets[chosen_segment + 1])
    row_ids = selection.sel_fids[start:end].to(t.int64).tolist()
    row_pcts = selection.sel_pcts[start:end].to(t.int64).tolist()
    row = dict(zip(row_ids, row_pcts, strict=True))
    assert row[first_pair[0]] == round(255 * float(chosen_pcts[:2].max().item()))
    assert row[second_pair[0]] == round(255 * float(chosen_pcts[2:].max().item()))

    # Run the exact identity-equivalent differential on this same complete
    # production shard. The active descriptor remains a distinct artifact
    # identity, but the post-mask representative map and negative universe are
    # literal identities, so the compressed shard bytes must match.
    mask = _real_mask()
    allowed = t.nonzero(mask.allowed_universe_mask(_N_SAE), as_tuple=False).squeeze(1)
    disallowed = _disallowed_ids(mask)
    identity_groups = [
        sorted([int(disallowed[index]), int(allowed[500 + index])]) for index in range(32)
    ]
    identity_clusters = _write_cluster_state(
        tmp_path,
        identity_groups,
        mask=mask,
        name="full_shard_identity_after_mask.json",
    )
    assert not bool(identity_clusters.consolidated_away.any())
    assert identity_clusters.is_identity_after_mask is True
    assert t.equal(
        identity_clusters.rep_of,
        t.arange(_N_SAE, dtype=t.int64),
    )
    assert identity_clusters.effective_negative_mask() == mask

    def identity_pipeline() -> FeaturePipelineConfig:
        return FeaturePipelineConfig(
            selection_config=ThresholdSelectionConfig(
                selection=ThresholdSelection(
                    ThresholdType.feature_percentile_segment_topk,
                    _PACKED_TOP8_THRESHOLD,
                ),
                precomputed_thresholds=t.zeros(_N_SAE, dtype=t.float32),
            ),
            boundaries=t.zeros(_N_SAE, 1, dtype=t.float64),
            negative_bundle=None,
            effective_buckets=2,
            feature_mask=mask,
        )

    unclustered_dir = tmp_path / "full_shard_unclustered"
    identity_dir = tmp_path / "full_shard_active_identity"
    unclustered_dir.mkdir()
    identity_dir.mkdir()
    selection_producer.select_features_for_shard(
        packed_path,
        identity_pipeline(),
        unclustered_dir,
        packed_path.stem.split(".")[0],
    )
    selection_producer.select_features_for_shard(
        packed_path,
        identity_pipeline(),
        identity_dir,
        packed_path.stem.split(".")[0],
        feature_clusters=identity_clusters,
    )
    unclustered_path = overlay_shard_path(
        unclustered_dir,
        packed_path.stem.split(".")[0],
    )
    identity_path = overlay_shard_path(
        identity_dir,
        packed_path.stem.split(".")[0],
    )
    assert identity_path.read_bytes() == unclustered_path.read_bytes()
    unclustered_selection = load_selection_shard(unclustered_path)
    identity_selection = load_selection_shard(identity_path)
    for name in (
        "sel_offsets",
        "sel_fids",
        "sel_buckets",
        "sel_pcts",
        "neg_offsets",
        "neg_fids",
    ):
        actual = getattr(identity_selection, name)
        expected = getattr(unclustered_selection, name)
        assert actual.dtype == expected.dtype, name
        assert t.equal(actual, expected), name
    identity_clusters.assert_selection_shard(
        sel_offsets=identity_selection.sel_offsets,
        sel_fids=identity_selection.sel_fids,
        sel_buckets=identity_selection.sel_buckets,
        sel_pcts=identity_selection.sel_pcts,
        neg_offsets=identity_selection.neg_offsets,
        neg_fids=identity_selection.neg_fids,
        expected_segments=packed.total_segments,
        effective_buckets=2,
        surface="full real identity-equivalent selection shard",
    )

    # Repeat the splice-specific exact differential at the real 65,536-feature
    # width and with the real resolved mask. Include mask-disallowed cluster
    # members in the raw counterfactual result. An inert-after-mask cluster uses
    # the exact raw legacy currency because its side consolidation is
    # mathematically the identity.
    raw_fids = t.tensor(
        [disallowed[0], allowed[500], disallowed[1], allowed[501]],
        dtype=t.int64,
    )
    raw_mags = t.tensor([20.0, 33.0, 50.0, 66.0], dtype=t.float32)
    raw_pcts = t.tensor([51, 84, 128, 168], dtype=t.uint8).float() / 255.0
    ref_fids = t.cat(
        [raw_fids, allowed[600:602].to(t.int64)],
    )
    ref_mags = t.tensor(
        [5.0, 5.0, 5.0, 5.0, 90.0, 80.0],
        dtype=t.float32,
    )
    ref_pcts = (
        t.tensor(
            [13, 13, 13, 13, 230, 204],
            dtype=t.uint8,
        ).float()
        / 255.0
    )
    delta_result = compute_doc_counterfactual_deltas(
        [raw_fids],
        [raw_mags],
        [raw_pcts],
        [(ref_fids, ref_mags, ref_pcts)],
        _N_SAE,
    )[0]
    assert delta_result is not None

    def identity_splice_pipeline() -> FeaturePipelineConfig:
        percentile_table = t.zeros((_N_SAE, 99), dtype=t.float32)
        boundaries = t.empty((_N_SAE, 2), dtype=t.float64)
        boundaries[:, 0] = 33.0
        boundaries[:, 1] = 66.0
        return FeaturePipelineConfig(
            selection_config=SamplingBundle(
                selection=SamplingSelection(
                    distribution=Distribution.normal,
                    mean=2.0,
                    std=0.01,
                    min_k=2,
                    max_k=3,
                    mode=SamplingMode.top_k_percentile,
                    delta_weights=DeltaSamplingWeights.percentile_delta,
                ),
                percentile_table=percentile_table,
                rng=t.Generator().manual_seed(714),
            ),
            boundaries=boundaries,
            negative_bundle=NegativeBundle(
                selection=NegativeSamplingSelection(
                    distribution=Distribution.normal,
                    mean=2.0,
                    std=0.01,
                    min_k=2,
                    max_k=3,
                ),
                n_sae_features=_N_SAE,
                rng=t.Generator().manual_seed(715),
                feature_mask=mask,
            ),
            effective_buckets=3,
            delta_bundle=DeltaBundle(
                config=FeatureDeltaConfig(
                    mode=DeltaMode.splice_counterfactual,
                    neg_delta_ratio=1.0,
                ),
                percentile_table=percentile_table,
            ),
            feature_mask=mask,
        )

    unclustered_splice_pipeline = identity_splice_pipeline()
    clustered_splice_pipeline = identity_splice_pipeline()
    unclustered_splice = _materialize_splice_segment_selection(
        raw_fids=raw_fids,
        raw_mags=raw_mags,
        raw_pcts=raw_pcts,
        delta_result=delta_result,
        pipeline_config=unclustered_splice_pipeline,
    )
    clustered_splice = _materialize_splice_segment_selection(
        raw_fids=raw_fids,
        raw_mags=raw_mags,
        raw_pcts=raw_pcts,
        delta_result=delta_result,
        pipeline_config=clustered_splice_pipeline,
        feature_clusters=identity_clusters,
    )
    assert clustered_splice == unclustered_splice
    assert isinstance(unclustered_splice_pipeline.selection_config, SamplingBundle)
    assert isinstance(clustered_splice_pipeline.selection_config, SamplingBundle)
    assert t.equal(
        clustered_splice_pipeline.selection_config.rng.get_state(),
        unclustered_splice_pipeline.selection_config.rng.get_state(),
    )
    assert unclustered_splice_pipeline.negative_bundle is not None
    assert clustered_splice_pipeline.negative_bundle is not None
    assert t.equal(
        clustered_splice_pipeline.negative_bundle.rng.get_state(),
        unclustered_splice_pipeline.negative_bundle.rng.get_state(),
    )
