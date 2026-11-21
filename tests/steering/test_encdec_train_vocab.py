"""The PRODUCTION model-construction path under compaction (#695 PR B).

`build_encdec_meta_model_from_training_cfg` is the function
`conditional_encdec_train.build_shared_state` actually calls. These tests drive it with
real Hydra-shaped configs, so a regression in the wiring — not just in the primitive —
fails here.

Also pins the compaction summary that the training script logs at startup: it is the
ONLY way an operator sees the realised compaction ratio, which depends on the corpus and
the frequency bounds and cannot be known offline.
"""

from __future__ import annotations

import inspect

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from errors import FeatureVocabError
from scripts.train.conditional_encdec_train import (
    _assert_compact_fp_checkpoint_compatible,
    build_encdec_meta_model_from_training_cfg,
    format_compaction_summary,
    no_compaction_reason,
    train_one_config,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.refined_types import PositiveInt

N_SAE: PositiveInt = PositiveInt.parse(64)
ALLOWED = (0, 3, 7, 10, 11, 20, 21, 30, 31, 40, 55, 63)
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)
MASK = FeatureMask(mode=FeatureMaskMode.whitelist, ids=ALLOWED)


def test_pretraining_restores_rank_state_and_gathers_next_preloaded_state() -> None:
    source = inspect.getsource(train_one_config)
    restore = source.index("select_dataloader_rank_state(")
    loader_build = source.index('logger.info("Building train dataloader...")')
    save_predicate = source.index("should_save_checkpoint = last_step or (")
    gather = source.index("build_dataloader_state_envelope(", save_predicate)
    metadata = source.index('"dataloader_state_dict": dataloader_state_envelope', gather)

    assert restore < loader_build
    assert save_predicate < gather < metadata
    assert "master_process" not in source[save_predicate:gather]


def _cfg(
    feature_prediction=None,
    *,
    depth: int = 2,
    aspect_ratio: int | float = 32,
    head_dim: int = 32,
) -> DictConfig:
    return OmegaConf.create(
        {
            "depth": depth,
            "aspect_ratio": aspect_ratio,
            "head_dim": head_dim,
            "max_seq_len": 16,
            "window_pattern": "L",
            "num_buckets": 4,
            "encoder": {
                "n_embd": 32, "depth": 2, "n_head": None, "n_kv_head": None, "head_dim": 16,
            },
            "cross_attention": {"frequency": 1, "n_head": None, "n_kv_head": None},
            "feature_prediction": feature_prediction,
        }
    )


class TestProductionConstruction:
    def test_fractional_aspect_ratio_builds_integer_decoder_shape(self):
        cfg = _cfg(depth=27, aspect_ratio=1664 / 27, head_dim=128)
        model = build_encdec_meta_model_from_training_cfg(
            cfg,
            base_vocab=128,
            n_sae_features=N_SAE,
            feature_vocab=None,
        )
        assert model.dec_config.n_embd == 1664
        assert model.dec_config.n_head == 13
        assert model.dec_config.n_kv_head == 13
        assert type(model.dec_config.n_embd) is int
        assert type(model.dec_config.n_head) is int

    def test_no_vocab_builds_a_full_width_encoder(self):
        m = build_encdec_meta_model_from_training_cfg(
            _cfg(), base_vocab=128, n_sae_features=N_SAE, feature_vocab=None
        )
        assert m.encoder.feature_embed.weight.shape[0] == N_SAE
        assert m.feature_vocab is None

    def test_vocab_compacts_every_feature_indexed_table(self):
        m = build_encdec_meta_model_from_training_cfg(
            _cfg(), base_vocab=128, n_sae_features=N_SAE, feature_vocab=VOCAB
        )
        assert m.encoder.feature_embed.weight.shape[0] == len(ALLOWED)
        for k in m.encoder.value_embeds:
            assert m.encoder.value_embeds[k].weight.shape[0] == len(ALLOWED)  # ty: ignore[not-subscriptable]

    def test_n_sae_features_stays_the_ORIGINAL_width(self):
        m = build_encdec_meta_model_from_training_cfg(
            _cfg(), base_vocab=128, n_sae_features=N_SAE, feature_vocab=VOCAB
        )
        assert m.encoder.config.n_sae_features == N_SAE
        # `feature_vocab_size`, NOT `vocab_size`: the latter means the TOKEN vocab
        # everywhere else in this repo (GPTConfig.vocab_size). See
        # test_encoder_feature_vocab.py::test_the_FEATURE_table_height_is_NOT_called_vocab_size
        assert m.feature_vocab_size == len(ALLOWED)

    def test_fp_head_plus_compaction_uses_the_effective_width(self):
        fp = {"block": "mlp", "head_layers": 0, "weight": 0.1, "layer": -1, "loss_mode": "best_token", "loss_fn": "bce"}
        model = build_encdec_meta_model_from_training_cfg(
            _cfg(fp), base_vocab=128, n_sae_features=N_SAE, feature_vocab=VOCAB
        )
        assert model.feature_pred_head is not None
        assert model.feature_pred_head.out_features == len(ALLOWED)
        assert model.feature_prediction_width == len(ALLOWED)

    def test_fp_head_without_compaction_still_builds(self):
        fp = {"block": "mlp", "head_layers": 0, "weight": 0.1, "layer": -1, "loss_mode": "best_token", "loss_fn": "bce"}
        m = build_encdec_meta_model_from_training_cfg(
            _cfg(fp), base_vocab=128, n_sae_features=N_SAE, feature_vocab=None
        )
        assert m.feature_pred_head is not None


class TestVocabFromMask:
    def test_blacklist_and_whitelist_both_produce_a_vocab(self):
        bl = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(1, 2, 3))
        wl = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 2, 3))
        assert FeatureVocab.from_mask(bl, N_SAE).size == N_SAE - 3  # ty: ignore[unresolved-attribute]
        assert FeatureVocab.from_mask(wl, N_SAE).size == 3  # ty: ignore[unresolved-attribute]

    def test_no_mask_means_no_compaction(self):
        assert FeatureVocab.from_mask(None, N_SAE) is None


class TestCompactFpCheckpointResumeIdentity:
    def _layout(self) -> FeaturePredictionTargetLayout:
        return FeaturePredictionTargetLayout.build(
            n_sae_features=N_SAE,
            feature_mask=MASK,
            feature_clusters=None,
            feature_vocab=VOCAB,
        )

    def _state(self) -> dict:
        return {
            "feature_vocab": VOCAB.metadata_dict(),
            "feature_prediction_width": VOCAB.size,
            "fp_target_source_certificate_sha256": "a" * 64,
        }

    def _model_data(self) -> dict:
        return {
            "feature_pred_head.weight": t.zeros(VOCAB.size, 32),
            "feature_pred_head.bias": t.zeros(VOCAB.size),
        }

    def test_complete_identity_and_head_shape_pass(self):
        _assert_compact_fp_checkpoint_compatible(
            self._layout(),
            "a" * 64,
            self._model_data(),
            {"dataloader_state_dict": self._state()},
            checkpoint_dir="/compact-fp",
        )

    def test_per_rank_envelope_selects_compact_identity(self):
        state = self._state()
        envelope = {
            "ddp_world_size": 2,
            "rank_states": [
                {**state, "ddp_world_size": 2, "batch_idx": 3},
                {**state, "ddp_world_size": 2, "batch_idx": 4},
            ],
        }
        _assert_compact_fp_checkpoint_compatible(
            self._layout(),
            "a" * 64,
            self._model_data(),
            {"dataloader_state_dict": envelope},
            checkpoint_dir="/compact-fp",
            ddp_rank=1,
            ddp_world_size=2,
        )

        with pytest.raises(ValueError, match=r"world.size.*2.*1|2.*1"):
            _assert_compact_fp_checkpoint_compatible(
                self._layout(),
                "a" * 64,
                self._model_data(),
                {"dataloader_state_dict": envelope},
                checkpoint_dir="/compact-fp",
                ddp_rank=0,
                ddp_world_size=1,
            )

    def test_partial_identity_changed_source_and_wrong_head_fail(self):
        partial = self._state()
        del partial["feature_prediction_width"]
        with pytest.raises(ValueError, match="missing/partial"):
            _assert_compact_fp_checkpoint_compatible(
                self._layout(),
                "a" * 64,
                self._model_data(),
                {"dataloader_state_dict": partial},
                checkpoint_dir="/compact-fp",
            )

        with pytest.raises(ValueError, match="different packed"):
            _assert_compact_fp_checkpoint_compatible(
                self._layout(),
                "b" * 64,
                self._model_data(),
                {"dataloader_state_dict": self._state()},
                checkpoint_dir="/compact-fp",
            )

        wrong_head = self._model_data()
        wrong_head["feature_pred_head.weight"] = t.zeros(VOCAB.size + 1, 32)
        with pytest.raises(FeatureVocabError, match="head width"):
            _assert_compact_fp_checkpoint_compatible(
                self._layout(),
                "a" * 64,
                wrong_head,
                {"dataloader_state_dict": self._state()},
                checkpoint_dir="/compact-fp",
            )

    def test_same_width_different_representative_ids_fail(self):
        other = FeatureVocab(N_SAE, tuple(range(VOCAB.size)))
        state = self._state()
        state["feature_vocab"] = other.metadata_dict()
        with pytest.raises(FeatureVocabError, match="vocabulary metadata"):
            _assert_compact_fp_checkpoint_compatible(
                self._layout(),
                "a" * 64,
                self._model_data(),
                {"dataloader_state_dict": state},
                checkpoint_dir="/compact-fp",
            )

    def test_compact_state_cannot_resume_full_or_headless(self):
        with pytest.raises(ValueError, match="full-width or has feature prediction disabled"):
            _assert_compact_fp_checkpoint_compatible(
                None,
                None,
                self._model_data(),
                {"dataloader_state_dict": self._state()},
                checkpoint_dir="/compact-fp",
            )


class TestCompactionSummary:
    """The startup log is the ONLY visibility into the realised compaction ratio.

    The vocabulary and the dictionary width are read OFF the model (#695 review finding
    14) — passing them alongside it let a caller describe a model that does not exist.
    Only the ABSENCE reason stays a parameter, because ``feature_vocab is None`` is two
    different worlds and the model cannot tell them apart (finding 9).
    """

    def test_summary_reports_rows_ratio_and_params_saved(self):
        m = build_encdec_meta_model_from_training_cfg(
            _cfg(), base_vocab=128, n_sae_features=N_SAE, feature_vocab=VOCAB
        )
        msg = format_compaction_summary(m, no_vocab_reason=no_compaction_reason(None, N_SAE))
        assert "12" in msg and "64" in msg  # V and n_sae
        assert "18.8%" in msg or "18.75%" in msg  # 12/64 kept
        # dropped rows * (enc_n_embd + n_ve*kv_dim) must be quoted exactly
        enc = m.encoder.config
        kv = enc.n_kv_head * enc.head_dim
        saved = (N_SAE - len(ALLOWED)) * (enc.n_embd + len(m.encoder.value_embeds) * kv)
        assert f"{saved:,}" in msg

    def test_summary_states_plainly_when_there_is_no_compaction(self):
        m = build_encdec_meta_model_from_training_cfg(
            _cfg(), base_vocab=128, n_sae_features=N_SAE, feature_vocab=None
        )
        msg = format_compaction_summary(m, no_vocab_reason=no_compaction_reason(None, N_SAE))
        assert "no compaction" in msg.lower()
        assert "feature_mask" in msg

    def test_summary_carries_NO_threshold_or_warning(self):
        """Owner decision: a plain factual log. No magic constant anywhere in this
        design — not one that changes behaviour, and not even one that changes what is
        printed."""
        m = build_encdec_meta_model_from_training_cfg(
            _cfg(), base_vocab=128, n_sae_features=N_SAE,
            feature_vocab=FeatureVocab(N_SAE, tuple(range(N_SAE - 1))),  # 98.4% kept
        )
        msg = format_compaction_summary(m, no_vocab_reason=no_compaction_reason(None, N_SAE))
        for banned in ("warn", "WARNING", "consider", "negligible", "too small"):
            assert banned not in msg, f"summary must stay factual; found {banned!r}"
