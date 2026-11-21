"""Real checkpoint/conditioning qualification for issue #1673.

This is deliberately a strict scheduled test: it consumes the exact audited
pod paths and fails when they are absent. It has no synthetic or alternate
checkpoint fallback. The ``slow`` marker keeps it out of ordinary CPU suites.
"""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import pytest
import torch as t

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from steering.conditional.encoder_decoder.eval_residual import (
    post_cross_attention_residual_counter,
)
from steering.eval.axbench_adv.counter import (
    ProjectCounterExpectations,
    preflight_project_direction_fit,
)
from steering.eval.axbench_adv.direction_fitting import (
    fit_project_directions,
)
from steering.eval.axbench_upstream.data import select_training_pairs
from steering.eval.axbench_upstream.methods import load_method_setup, load_target_model

PREPARED_DIR = Path("/workspace/stage/axbench_inputs/prepared")
TARGET_DIR = Path("/workspace/axbench/target_leaf_s32mask")
TARGET_META = TARGET_DIR / "meta_001125.json"
TARGET_MODEL = TARGET_DIR / "model_001125.pt"
STATISTICS = Path("/workspace/stage/stats/feature_statistics.safetensors")
EXPECTED_NANOCHAT_BASE = Path("/workspace/nanochat")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cache(model) -> KVCache:
    config = model.dec_config
    return KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=config.sequence_len,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device=model.get_device(),
        dtype=COMPUTE_DTYPE,
    )


@pytest.mark.slow
def test_real_masked_s32_fit_artifact_and_active_prefill_decode(tmp_path: Path) -> None:
    assert t.cuda.is_available(), "scheduled issue-1673 qualification requires CUDA"
    assert Path(os.environ.get("NANOCHAT_BASE_DIR", "")).resolve() == EXPECTED_NANOCHAT_BASE
    for path in (PREPARED_DIR, TARGET_META, TARGET_MODEL, STATISTICS):
        assert path.exists(), f"exact issue-1673 qualification input is missing: {path}"

    output_root = tmp_path / "issue1673-directions"
    publication = fit_project_directions(
        prepared_dir=PREPARED_DIR,
        target_meta_path=TARGET_META,
        feature_statistics_path=STATISTICS,
        output_root=output_root,
        max_concepts=1,
        layer=21,
        seed=42,
        device="cuda:0",
    )
    assert len(publication.artifacts) == 2

    setup = load_method_setup(
        TARGET_META,
        STATISTICS,
        require_exact_statistics_identity=True,
    )
    training = select_training_pairs(PREPARED_DIR, max_concepts=1)
    model_config = setup.meta["model_config"]
    preflight = preflight_project_direction_fit(
        output_root,
        ProjectCounterExpectations(
            target_model_sha256=_sha256(TARGET_MODEL),
            target_metadata_sha256=_sha256(TARGET_META),
            concept_texts={0: training[0].concept},
            decoder_width=model_config["n_embd"],
            layer=21,
            fit_seed=42,
        ),
    )
    direction = preflight.artifacts[("diffmean", 0)].loaded.direction

    gc.collect()
    t.cuda.empty_cache()
    model, _tokenizer = load_target_model(setup, "cuda:0")
    model.eval()
    state_keys = tuple(model.state_dict())
    original_id = (
        0 if setup.feature_vocab is None else int(setup.feature_vocab.original_ids[0])
    )
    tokens = t.tensor([[1, 2, 3, 4]], device="cuda:0", dtype=t.int64)
    feature_ids = t.tensor([[[original_id]]], device="cuda:0", dtype=t.int64)
    magnitudes = t.zeros_like(feature_ids)
    mask = t.ones_like(feature_ids, dtype=t.bool)
    seg_map = t.zeros_like(tokens)

    def forward(
        current_tokens: t.Tensor,
        current_seg_map: t.Tensor,
        *,
        cache: KVCache | None = None,
        return_fp_logits: bool = False,
        fp_logits_only: bool = False,
    ):
        return model(
            current_tokens,
            None,
            enc_feature_ids=feature_ids,
            enc_magnitudes=magnitudes,
            enc_mask=mask,
            seg_map=current_seg_map,
            encoder_strict=True,
            kv_cache=cache,
            return_fp_logits=return_fp_logits,
            fp_logits_only=fp_logits_only,
        )

    with t.inference_mode():
        default_logits, default_fp = forward(tokens, seg_map, return_fp_logits=True)
        with post_cross_attention_residual_counter(
            model,
            layer=21,
            direction=direction,
            magnitude=0.0,
        ):
            zero_logits, zero_fp = forward(tokens, seg_map, return_fp_logits=True)
        assert t.equal(default_logits, zero_logits)
        assert t.equal(default_fp, zero_fp)

        for signed_direction in (direction, -direction):
            with post_cross_attention_residual_counter(
                model,
                layer=21,
                direction=signed_direction,
                magnitude=0.25,
            ):
                active_logits, active_fp = forward(tokens, seg_map, return_fp_logits=True)
                assert not t.equal(active_logits, default_logits)
                assert not t.equal(active_fp, default_fp)
                cache = _cache(model)
                forward(tokens[:, :3], seg_map[:, :3], cache=cache)
                decoded_fp = forward(
                    tokens[:, 3:],
                    seg_map[:, 3:],
                    cache=cache,
                    fp_logits_only=True,
                )
                assert decoded_fp.shape == (1, 1, model.feature_prediction_width)
            assert model._post_cross_attention_eval_callback is None
    assert tuple(model.state_dict()) == state_keys
