"""build_encdec_model must be vocab-aware (#695 PR B).

``checkpoint_manager.py:523`` infers ``n_sae_features`` from
``encoder.feature_embed.weight.shape[0]``. Once the table is compacted that row count
is **V**, not ``n_sae`` — so the loader would silently redefine ``n_sae_features`` for
every downstream consumer (``eval_adapter``, ``target_conditioning``,
``model_detection``, ``harmfulness_eval``, all of which use it to bounds-check
ORIGINAL feature ids).

These tests exercise the REAL loader against REAL checkpoints written to disk. Nothing
is mocked.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch as t

from errors import EvalConditioningError, FeatureVocabError
from nanochat.checkpoint_manager import (
    ModelPhase,
    build_encdec_model,
    load_checkpoint,
    save_checkpoint,
)
from nanochat.gpt import GPTConfig
from scripts.train.conditional_encdec_train import (
    _assert_not_pre_effective_compaction_checkpoint,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    ResolvedFeatureClusters,
    assert_resume_feature_clusters_match,
    feature_clusters_checkpoint_metadata,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    canonical_dict,
)
from steering.conditional.feature_vocab import (
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
    FeatureVocab,
    assert_checkpoint_vocab_compatible,
    assert_resume_feature_vocab_matches,
)


@pytest.fixture(autouse=True)
def _fake_tokenizer():
    """build_encdec_model asserts the tokenizer's vocab size matches the checkpoint's.

    The tokenizer is entirely orthogonal to the feature-vocabulary logic under test,
    and the real tokenizer.pkl lives outside this worktree. Patching it is the repo's
    established convention (see test_checkpoint_manager_encdec_autodetect.py:95).
    """
    tok = Mock()
    tok.get_vocab_size.return_value = 65536
    with patch("nanochat.checkpoint_manager.get_tokenizer", return_value=tok):
        yield


@pytest.fixture
def ckpt(tmp_path_factory) -> Path:
    """A checkpoint dir whose PATH DOES NOT CONTAIN THE TEST'S OWN NAME.

    Not cosmetic — it closes a whole class of false passes.

    Every #695 guard embeds the checkpoint dir in its message (deliberately: the operator
    must be told WHICH checkpoint). pytest's ``tmp_path`` is named after the test. So a
    ``pytest.raises(..., match="sorted")`` in a test called
    ``test_an_UNSORTED_ids_buffer_raises`` matches THE TEST'S OWN DIRECTORY NAME — and
    passes even with the unsorted-ids guard deleted outright. Measured, not theorised:
    with ``feature_vocab_from_state_dict`` reverted to the pre-#695 shape-inference, that
    test still went green (a *different* guard raised, and the regex hit the path).

    ``tmp_path_factory.mktemp`` gives a fixed-prefix dir instead, so a ``match=`` can only
    be satisfied by the guard's actual wording.
    """
    return tmp_path_factory.mktemp("encdec_ckpt")


N_SAE = 64
ALLOWED = (0, 3, 7, 10, 11, 20, 21, 30, 31, 40, 55, 63)
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)

DEC_KW = {
    "sequence_len": 16,
    "vocab_size": 65536,
    "n_layer": 2,
    "n_head": 2,
    "n_kv_head": 2,
    "n_embd": 64,
    "window_pattern": "L",
}
ENC_KW = {
    "n_embd": 32,
    "depth": 2,
    "n_head": None,
    "n_kv_head": None,
    "head_dim": 16,
}


def _build(
    vocab: FeatureVocab | None,
    seed: int = 3,
    *,
    feature_pred_config: FeaturePredictionConfig | None = None,
) -> EncoderDecoderGPT:
    with t.device("meta"):
        m = EncoderDecoderGPT(
            GPTConfig(
                sequence_len=16,
                vocab_size=65536,
                n_layer=2,
                n_head=2,
                n_kv_head=2,
                n_embd=64,
                window_pattern="L",
            ),
            EncoderConfig(
                n_embd=32,
                depth=2,
                n_head=None,
                n_kv_head=None,
                head_dim=16,
            ),
            CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
            n_sae_features=N_SAE,
            num_buckets=4,
            feature_pred_config=feature_pred_config,
            feature_vocab=vocab,
        )
    m.to_empty(device="cpu")
    t.manual_seed(seed)
    m.init_weights()
    with t.no_grad():  # de-degenerate so logits are a real function of every weight
        g = t.Generator().manual_seed(seed + 1)
        for p in m.parameters():
            p.copy_(t.randn(p.shape, generator=g) * 0.05)
    m.encoder.init_vocab_buffers()  # restore buffers clobbered by the randomisation
    return m


def _save(
    ckpt,
    model: EncoderDecoderGPT,
    step: int = 5,
    *,
    extra_meta: dict | None = None,
) -> str:
    user_config: dict[str, object] = {
        "encoder": dict(ENC_KW),
        "cross_attention": {"frequency": 1, "n_head": None, "n_kv_head": None},
        "num_buckets": 4,
    }
    meta: dict[str, object] = {
        "step": step,
        "model_config": dict(DEC_KW),
        "user_config": user_config,
    }
    if model.feature_vocab is not None:
        meta["feature_vocab"] = model.feature_vocab.metadata_dict()
    if model.feature_pred_config is not None:
        user_config["feature_prediction"] = {
            "block": "mlp",
            "head_layers": 0,
            "weight": model.feature_pred_config.weight,
            "layer": model.feature_pred_config.layer,
            "loss_mode": model.feature_pred_config.loss_mode.value,
            "loss_fn": model.feature_pred_config.loss_fn.value,
        }
    if extra_meta is not None:
        meta.update(extra_meta)
    save_checkpoint(str(ckpt), step, model.state_dict(), None, meta, rank=0)
    return str(ckpt)


def _fwd(model: EncoderDecoderGPT, feature_ids: tuple[int, int, int] = (10, 20, 30)) -> t.Tensor:
    model.eval()
    idx = t.arange(16).unsqueeze(0) % 128
    seg_map = (t.arange(16) // 8).unsqueeze(0)
    fids = t.zeros(1, 2, 5, dtype=t.int64)
    mags = t.zeros(1, 2, 5, dtype=t.int64)
    mask = t.zeros(1, 2, 5, dtype=t.bool)
    for s in range(2):
        fids[0, s, :3] = t.tensor(feature_ids)
        mags[0, s, :3] = t.tensor([1, 2, 3])
        mask[0, s, :3] = True
    with t.no_grad():
        return model(idx, None, fids, mags, mask, seg_map)


# --- tamper helpers: corrupt a REAL checkpoint on disk ----------------------


def _tamper_sd(ckpt_dir: Path, mutate: Callable[[dict], None], step: int = 5) -> str:
    p = ckpt_dir / f"model_{step:06d}.pt"
    sd = t.load(p, weights_only=True)
    mutate(sd)
    t.save(sd, p)
    return str(ckpt_dir)


def _tamper_meta(ckpt_dir: Path, mutate: Callable[[dict], None], step: int = 5) -> str:
    p = ckpt_dir / f"meta_{step:06d}.json"
    meta = json.loads(p.read_text())
    mutate(meta)
    p.write_text(json.dumps(meta))
    return str(ckpt_dir)


def _resume_into(model: EncoderDecoderGPT, ckpt_dir: str, step: int = 5) -> EncoderDecoderGPT:
    """The unclustered subset of the trainer's exact resume guard/load sequence.

    Driving the real functions in the real order matters: a guard that is correct but
    called AFTER ``load_state_dict`` would let PyTorch's bare ``size mismatch`` win, and
    one that is never called at all would let a wrong-mask resume through silently. Unit
    tests on synthetic dicts cannot see either mistake.
    """
    model_data, _, meta_data = load_checkpoint(ckpt_dir, step, "cpu", load_optimizer=False, rank=0)
    assert_checkpoint_vocab_compatible(model.feature_vocab, model_data, checkpoint_dir=ckpt_dir)
    model.load_state_dict(model_data, strict=True, assign=True)
    model.assert_vocab_consistent()
    assert_resume_feature_vocab_matches(model.feature_vocab, meta_data, checkpoint_dir=ckpt_dir)
    return model


def _cluster_state(
    root: Path,
) -> tuple[ResolvedFeatureClusters, FeatureClusterConsolidation, FeatureVocab]:
    path = root / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clusters": [[3, 7, 10], [20, 21]],
            }
        ),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        N_SAE,
    )
    assert resolved is not None
    consolidation = FeatureClusterConsolidation.build(resolved, MASK_A)
    vocab = effective_feature_vocab(consolidation, N_SAE)
    assert vocab is not None
    return resolved, consolidation, vocab


def _cluster_meta(resolved: ResolvedFeatureClusters) -> dict:
    return {
        "feature_mask_canonical": canonical_dict(MASK_A),
        **feature_clusters_checkpoint_metadata(resolved),
    }


def _resume_into_clustered(
    model: EncoderDecoderGPT,
    ckpt_dir: str,
    *,
    resolved: ResolvedFeatureClusters,
    consolidation: FeatureClusterConsolidation,
    step: int = 5,
) -> EncoderDecoderGPT:
    """The cluster-aware trainer sequence, including the migration diagnostic."""
    model_data, _, meta_data = load_checkpoint(
        ckpt_dir,
        step,
        "cpu",
        load_optimizer=False,
        rank=0,
    )
    assert_resume_feature_clusters_match(
        resolved,
        meta_data,
        checkpoint_dir=ckpt_dir,
    )
    _assert_not_pre_effective_compaction_checkpoint(
        model.feature_vocab,
        MASK_A,
        consolidation,
        model_data,
        checkpoint_dir=ckpt_dir,
    )
    assert_checkpoint_vocab_compatible(
        model.feature_vocab,
        model_data,
        checkpoint_dir=ckpt_dir,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    model.assert_vocab_consistent()
    assert_resume_feature_vocab_matches(
        model.feature_vocab,
        meta_data,
        checkpoint_dir=ckpt_dir,
    )
    return model


# Mask A (what the checkpoint was trained with) and the ways a resume can differ from it.
MASK_A = FeatureMask(mode=FeatureMaskMode.whitelist, ids=ALLOWED)
MASK_SAME_SIZE = FeatureMask(
    mode=FeatureMaskMode.whitelist, ids=(1, 4, 8, 12, 13, 22, 23, 32, 33, 41, 56, 62)
)
MASK_SUBSET = FeatureMask(mode=FeatureMaskMode.whitelist, ids=ALLOWED[:-1])
MASK_SUPERSET = FeatureMask(mode=FeatureMaskMode.whitelist, ids=ALLOWED + (5,))
# The blacklist that allows EXACTLY the same universe as MASK_A.
MASK_A_AS_BLACKLIST = FeatureMask(
    mode=FeatureMaskMode.blacklist,
    ids=tuple(i for i in range(N_SAE) if i not in ALLOWED),
)


class TestCompactedCheckpointRoundTrip:
    def test_compact_fp_head_round_trips_through_vendored_loader(self, ckpt):
        fp = FeaturePredictionConfig(weight=0.1, layer=-1)
        model = _build(VOCAB, feature_pred_config=fp)
        checkpoint_dir = _save(ckpt, model)

        loaded, _, _ = build_encdec_model(
            checkpoint_dir,
            5,
            "cpu",
            ModelPhase.TRAIN,
            eval_conditioning=EvalConditioning.not_evaluating(),
        )

        assert loaded.feature_pred_head is not None
        assert loaded.feature_pred_head.out_features == VOCAB.size
        assert loaded.feature_prediction_width == VOCAB.size
        for key, expected in model.state_dict().items():
            assert t.equal(loaded.state_dict()[key], expected), key

    def test_n_sae_features_is_the_ORIGINAL_width_not_V(self, ckpt):
        """THE trap. feature_embed has V=12 rows; n_sae_features must still be 64."""
        d = _save(ckpt, _build(VOCAB))
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        assert loaded.encoder.feature_embed.weight.shape[0] == len(ALLOWED) == 12
        assert loaded.encoder.config.n_sae_features == N_SAE == 64

    def test_vocab_is_reconstructed_from_the_FILE_ALONE(self, ckpt):
        """No training config, no sidecar (requirement 1)."""
        d = _save(ckpt, _build(VOCAB))
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        assert loaded.feature_vocab == VOCAB

    def test_logits_survive_the_round_trip_bitwise(self, ckpt):
        m = _build(VOCAB)
        before = _fwd(m)
        d = _save(ckpt, m)
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        assert t.equal(before, _fwd(loaded))

    def test_loaded_model_passes_its_own_consistency_check(self, ckpt):
        d = _save(ckpt, _build(VOCAB))
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        loaded.assert_vocab_consistent()

    def test_the_remap_is_correct_after_load(self, ckpt):
        """load_state_dict(assign=True) replaces the persistent feature_vocab_ids buffer with
        the checkpoint's. ``_to_compact`` derives the remap from that buffer per-forward (no
        stored map), so every ORIGINAL id must map to its compact row after a round trip."""
        d = _save(ckpt, _build(VOCAB))
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        ids = t.tensor([list(ALLOWED)], dtype=t.int64)
        got = loaded.encoder._to_compact(ids, t.ones(1, len(ALLOWED), dtype=t.bool))
        assert got.tolist() == [list(range(len(ALLOWED)))]


class TestClusteredEffectiveCheckpointRoundTrip:
    def test_effective_vocab_round_trips_through_real_save_and_loader(self, ckpt):
        resolved, consolidation, vocab = _cluster_state(ckpt)
        assert vocab.size == VOCAB.size - int(consolidation.consolidated_away.sum().item())
        model = _build(vocab)
        expected = _fwd(model, feature_ids=(3, 20, 30))
        checkpoint_dir = _save(
            ckpt,
            model,
            extra_meta=_cluster_meta(resolved),
        )

        loaded, _, meta = build_encdec_model(
            checkpoint_dir,
            5,
            "cpu",
            ModelPhase.TRAIN,
            eval_conditioning=EvalConditioning.not_evaluating(),
        )

        assert loaded.feature_vocab == vocab
        assert loaded.encoder.feature_vocab_ids.tolist() == list(vocab.original_ids)
        assert meta["feature_vocab"] == vocab.metadata_dict()
        assert meta["feature_clusters_descriptor"] == resolved.descriptor_dict()
        loaded.assert_vocab_consistent()
        assert t.equal(expected, _fwd(loaded, feature_ids=(3, 20, 30)))

    def test_same_effective_vocab_resumes_through_cluster_aware_helper(self, ckpt):
        resolved, consolidation, vocab = _cluster_state(ckpt)
        trained = _build(vocab, seed=3)
        expected = _fwd(trained, feature_ids=(3, 20, 30))
        checkpoint_dir = _save(
            ckpt,
            trained,
            extra_meta=_cluster_meta(resolved),
        )
        fresh = _build(vocab, seed=99)

        _resume_into_clustered(
            fresh,
            checkpoint_dir,
            resolved=resolved,
            consolidation=consolidation,
        )

        assert t.equal(expected, _fwd(fresh, feature_ids=(3, 20, 30)))

    def test_inert_clusters_resume_cleanly_through_the_migration_diagnostic(self, ckpt):
        """The diagnostic's `checkpoint_vocab != configured_vocab` clause, exercised.

        Inert clusters (every cluster ≤1 mask-surviving member) are the ONE state where
        the mask-only and effective vocabularies coincide, so the diagnostic's first two
        conditions are BOTH true and only the inequality clause stands between a legal
        resume and a spurious "written by pre-effective-compaction code" refusal. A
        mutant dropping that clause fails here — previously nothing exercised it.
        """
        path = ckpt / "inert_clusters.json"
        path.write_text(
            json.dumps({"schema_version": 1, "clusters": [[3, 5, 6]]}),
            encoding="utf-8",
        )
        resolved = resolve_feature_clusters(
            {"feature_clusters": {"path": str(path)}},
            N_SAE,
        )
        assert resolved is not None
        consolidation = FeatureClusterConsolidation.build(resolved, MASK_A)
        assert consolidation.is_identity_after_mask  # 5 and 6 are mask-excluded
        vocab = effective_feature_vocab(consolidation, N_SAE)
        assert vocab == VOCAB  # inert ⇒ coincides with the mask-only vocabulary

        trained = _build(vocab, seed=3)
        expected = _fwd(trained, feature_ids=(3, 20, 30))
        checkpoint_dir = _save(ckpt, trained, extra_meta=_cluster_meta(resolved))
        fresh = _build(vocab, seed=99)

        _resume_into_clustered(
            fresh,
            checkpoint_dir,
            resolved=resolved,
            consolidation=consolidation,
        )

        assert t.equal(expected, _fwd(fresh, feature_ids=(3, 20, 30)))

    def test_pre_effective_checkpoint_gets_dedicated_resume_diagnostic(self, ckpt):
        resolved, consolidation, effective_vocab = _cluster_state(ckpt)
        legacy_mask_only = FeatureVocab.from_mask(MASK_A, N_SAE)
        assert legacy_mask_only == VOCAB
        checkpoint_dir = _save(
            ckpt,
            _build(legacy_mask_only),
            extra_meta=_cluster_meta(resolved),
        )

        with pytest.raises(FeatureVocabError) as exc:
            _resume_into_clustered(
                _build(effective_vocab),
                checkpoint_dir,
                resolved=resolved,
                consolidation=consolidation,
            )
        message = str(exc.value)
        assert "pre-effective-compaction code" in message
        assert "consolidated-away cluster members" in message
        assert "START FRESH" in message
        assert "no tool" in message

    def test_generic_vocab_mismatch_text_names_cluster_causes(self, ckpt):
        resolved, _, effective_vocab = _cluster_state(ckpt)
        checkpoint_dir = _save(
            ckpt,
            _build(VOCAB),
            extra_meta=_cluster_meta(resolved),
        )
        model_data, _, _ = load_checkpoint(
            checkpoint_dir,
            5,
            "cpu",
            load_optimizer=False,
            rank=0,
        )
        different = FeatureVocab(N_SAE, effective_vocab.original_ids[:-1])

        with pytest.raises(FeatureVocabError) as exc:
            assert_checkpoint_vocab_compatible(
                different,
                model_data,
                checkpoint_dir=checkpoint_dir,
            )
        message = str(exc.value)
        assert "same mask with different clusters" in message
        assert "before cluster-aware compaction" in message


class TestEvalOnACompactedCheckpointIsDeferred:
    """Evaluating a compacted checkpoint is deferred, and now via an explicit, per-eval
    ``EvalConditioning`` stance (#695 load-time contract) rather than #703's blanket guard.

    The load itself works for TRAIN (resume) and for inspection regardless of stance; only
    ``ModelPhase.EVAL`` on an encoder-decoder build consults the stance. A ``deferred(eval_name)``
    stance is a no-op on a legacy (uncompacted) checkpoint and raises ``NotImplementedError`` on a
    compacted one; a ``not_evaluating()`` stance reaching this path at all is itself a caller
    mis-declaration and raises ``EvalConditioningError`` — training/resume never reaches here.
    """

    def test_compacted_EVAL_deferred_raises_NotImplementedError(self, ckpt):
        d = _save(ckpt, _build(VOCAB))
        with pytest.raises(NotImplementedError, match="COMPACTED encoder-decoder checkpoint"):
            build_encdec_model(
                d,
                5,
                "cpu",
                ModelPhase.EVAL,
                eval_conditioning=EvalConditioning.deferred("unit_test_eval"),
            )

    def test_the_deferral_message_names_the_eval_and_dump_feature_mask(self, ckpt):
        d = _save(ckpt, _build(VOCAB))
        with pytest.raises(NotImplementedError) as exc:
            build_encdec_model(
                d,
                5,
                "cpu",
                ModelPhase.EVAL,
                eval_conditioning=EvalConditioning.deferred("unit_test_eval"),
            )
        assert "unit_test_eval" in str(exc.value) and "dump_feature_mask" in str(exc.value)

    def test_compacted_EVAL_not_evaluating_raises_EvalConditioningError(self, ckpt):
        d = _save(ckpt, _build(VOCAB))
        with pytest.raises(EvalConditioningError, match="not_evaluating"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.EVAL, eval_conditioning=EvalConditioning.not_evaluating()
            )

    def test_compacted_TRAIN_loads_regardless_of_stance(self, ckpt):
        d = _save(ckpt, _build(VOCAB))
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        assert loaded.feature_vocab is not None and loaded.feature_vocab.size == len(ALLOWED)

    def test_uncompacted_EVAL_deferred_LOADS(self, ckpt):
        d = _save(ckpt, _build(None))
        loaded, _, _ = build_encdec_model(
            d,
            5,
            "cpu",
            ModelPhase.EVAL,
            eval_conditioning=EvalConditioning.deferred("unit_test_eval"),
        )
        assert loaded.feature_vocab is None

    def test_eval_conditioning_is_a_required_keyword(self, ckpt):
        """Permanent regression test for the #695 load-time contract (Task 4): the transient
        None-default + 'unmigrated-load' coercion is gone, and ``eval_conditioning`` is now a
        required keyword-only argument. A call that omits it must not silently fall back to any
        stance — it must not even reach the function body. This is the non-vacuity proof of
        "required": invalid state (an undeclared stance) is unrepresentable, not merely
        runtime-checked."""
        with pytest.raises(TypeError, match="eval_conditioning"):
            inspect.signature(build_encdec_model).bind(
                str(ckpt), 5, "cpu", ModelPhase.EVAL
            )


class TestLegacyCheckpointStillLoads:
    def test_uncompacted_checkpoint_loads_as_identity(self, ckpt):
        """Requirement 5: existing checkpoints carry no vocab and must load unchanged."""
        d = _save(ckpt, _build(None))
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.EVAL, eval_conditioning=EvalConditioning.deferred("test")
        )
        assert loaded.feature_vocab is None
        assert loaded.encoder.config.n_sae_features == N_SAE
        assert loaded.encoder.feature_embed.weight.shape[0] == N_SAE

    def test_uncompacted_logits_survive_the_round_trip_bitwise(self, ckpt):
        m = _build(None)
        before = _fwd(m)
        d = _save(ckpt, m)
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        assert t.equal(before, _fwd(loaded))

    def test_legacy_state_dict_has_NO_vocab_keys(self, ckpt):
        """A no-mask run must produce a byte-identical state_dict to pre-#695, or every
        existing strict=True load breaks."""
        _save(ckpt, _build(None))
        sd = t.load(ckpt / "model_000005.pt", weights_only=True)
        assert not any("feature_vocab" in k for k in sd)


class TestMetaMirrorIsAMirrorNotASidecar:
    """ABSENCE of the meta mirror is fine. DISAGREEMENT is not.

    Requirement 1 says a loaded model is self-describing and "does not need the training
    config or a sidecar file to interpret its own embeddings" — the persistent buffers ARE
    the record. The meta ``feature_vocab`` block is a cheap FINGERPRINT so tooling can
    compare a checkpoint's vocabulary without torch.load-ing a multi-GB .pt.

    Making it MANDATORY at load would mean the sidecar IS required, i.e. requirement 1 is
    not delivered — and it would contradict ``scripts/analysis/dump_feature_mask.py``,
    which documents in this same PR that "ABSENCE is fine. Its *disagreement* is not."
    That tool is the one every #695 error message tells the operator to run, so a thinned
    meta must not be refused by the loader while the recovery tool happily reads it.

    (This file is also full of ``_recover_model_config`` / ``_iter_ancestor_metas``
    precisely because SFT/eval metas routinely omit fields.)
    """

    def test_compacted_checkpoint_with_NO_meta_mirror_still_loads_from_the_buffers(self, ckpt):
        m = _build(VOCAB)
        d = _save(ckpt, m)
        meta_path = ckpt / "meta_000005.json"
        meta = json.loads(meta_path.read_text())
        del meta["feature_vocab"]  # thinned meta — the weights still describe themselves
        meta_path.write_text(json.dumps(meta))

        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        assert loaded.feature_vocab == VOCAB
        assert loaded.encoder.config.n_sae_features == N_SAE
        loaded.assert_vocab_consistent()

    def test_and_its_logits_are_still_bitwise_correct(self, ckpt):
        m = _build(VOCAB)
        before = _fwd(m)
        d = _save(ckpt, m)
        meta_path = ckpt / "meta_000005.json"
        meta = json.loads(meta_path.read_text())
        del meta["feature_vocab"]
        meta_path.write_text(json.dumps(meta))

        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        assert t.equal(before, _fwd(loaded))

    def test_but_a_meta_claiming_a_vocab_the_weights_do_NOT_have_still_raises(self, ckpt):
        """The cross-check the owner asked for, intact: a REAL disagreement."""
        m = _build(None)  # uncompacted weights
        d = _save(ckpt, m)
        meta_path = ckpt / "meta_000005.json"
        meta = json.loads(meta_path.read_text())
        meta["feature_vocab"] = VOCAB.metadata_dict()  # meta lies
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(FeatureVocabError, match="disagrees with the weights"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )


class TestCorruptCheckpointsFailLoud:
    def test_meta_vocab_disagreeing_with_the_weights_raises(self, ckpt):
        """meta is a mirror, the buffers are the truth. If they disagree, the file is
        corrupt and we must not guess which one is right."""
        m = _build(VOCAB)
        d = _save(ckpt, m)
        meta_path = ckpt / "meta_000005.json"
        meta = json.loads(meta_path.read_text())
        meta["feature_vocab"]["sha256"] = "0" * 64  # tamper
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(FeatureVocabError, match="disagrees with the weights"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )

    def test_half_a_vocab_in_the_state_dict_raises(self, ckpt):
        d = _save(ckpt, _build(VOCAB))
        p = ckpt / "model_000005.pt"
        sd = t.load(p, weights_only=True)
        del sd["encoder.feature_vocab_n_sae"]  # keep ids, drop the width
        t.save(sd, p)
        with pytest.raises(FeatureVocabError, match="must carry both vocab buffers"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )


class TestEveryCorruptionFailsLoudThroughTheREALLoader:
    """The same corruptions ``test_feature_vocab_checkpoint.py`` drives through
    ``feature_vocab_from_state_dict`` on synthetic dicts, driven instead through
    ``build_encdec_model`` on a REAL ``.pt`` written to disk.

    A pure-function guard proves nothing on its own: it can be correct and still be
    called too late, called on the wrong dict, or have its exception swallowed. These
    pin that the LOADER actually refuses each file.
    """

    def test_the_other_half_of_a_vocab_raises(self, ckpt):
        """The mirror of ``test_half_a_vocab_in_the_state_dict_raises``: ids dropped,
        width kept. Both directions, or the ``has_ids != has_n_sae`` XOR is half-tested."""
        _save(ckpt, _build(VOCAB))
        d = _tamper_sd(ckpt, lambda sd: sd.pop(VOCAB_IDS_KEY))
        with pytest.raises(FeatureVocabError, match="must carry both vocab buffers"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )

    def test_an_id_at_or_above_n_sae_raises(self, ckpt):
        """An id that is not an SAE feature id at all — it must be rejected as out of range
        rather than silently searchsorted into a compact row."""
        _save(ckpt, _build(VOCAB))
        d = _tamper_sd(
            ckpt,
            lambda sd: sd.__setitem__(
                VOCAB_IDS_KEY, t.tensor(ALLOWED[:-1] + (N_SAE,), dtype=t.int64)
            ),
        )
        with pytest.raises(FeatureVocabError, match="is out of range"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )

    def test_a_negative_id_raises(self, ckpt):
        """``-1`` is the OUT-OF-VOCAB SENTINEL. A ``-1`` that reaches the vocab as a real
        id would make ``_to_compact``'s ``compact < 0`` guard indistinguishable from a
        legitimate row — the exact ambiguity the poisoned-buffer design exists to avoid."""
        _save(ckpt, _build(VOCAB))
        d = _tamper_sd(
            ckpt,
            lambda sd: sd.__setitem__(VOCAB_IDS_KEY, t.tensor((-1,) + ALLOWED[1:], dtype=t.int64)),
        )
        with pytest.raises(FeatureVocabError, match="must be non-negative"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )

    def test_an_UNSORTED_ids_buffer_raises_rather_than_being_silently_re_sorted(self, ckpt):
        """The buffer's ORDER *is* the row mapping. ``FeatureVocab.__post_init__``
        canonicalises, so an unsorted buffer would be silently re-sorted and compact row j
        would then name a DIFFERENT feature than the weights encode — no error anywhere."""
        _save(ckpt, _build(VOCAB))
        d = _tamper_sd(
            ckpt,
            lambda sd: sd.__setitem__(
                VOCAB_IDS_KEY, t.tensor(tuple(reversed(ALLOWED)), dtype=t.int64)
            ),
        )
        with pytest.raises(FeatureVocabError, match="is not sorted ascending"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )

    def test_a_DUPLICATED_id_raises(self, ckpt):
        """De-duplication would silently shrink V below the table's row count."""
        _save(ckpt, _build(VOCAB))
        dupes = ALLOWED[:-1] + (ALLOWED[-2],)  # ...55, 55 — still ascending!
        d = _tamper_sd(
            ckpt, lambda sd: sd.__setitem__(VOCAB_IDS_KEY, t.tensor(dupes, dtype=t.int64))
        )
        with pytest.raises(FeatureVocabError, match="duplicate feature ids"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )

    def test_V_disagreeing_with_the_feature_embed_row_count_raises(self, ckpt):
        """``feature_embed`` has V rows BY DEFINITION. If it doesn't, the file is corrupt
        and we must not pick a winner."""
        _save(ckpt, _build(VOCAB))
        d = _tamper_sd(
            ckpt,
            lambda sd: sd.__setitem__(
                VOCAB_IDS_KEY,
                t.tensor(ALLOWED[:-2], dtype=t.int64),  # V=10, table has 12
            ),
        )
        with pytest.raises(FeatureVocabError, match="rows but the vocab declares"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )

    def test_an_n_sae_buffer_of_zero_raises(self, ckpt):
        _save(ckpt, _build(VOCAB))
        d = _tamper_sd(ckpt, lambda sd: sd.__setitem__(VOCAB_N_SAE_KEY, t.tensor(0, dtype=t.int64)))
        with pytest.raises(FeatureVocabError, match="must be a positive int"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )

    def test_value_embeds_row_count_disagreeing_with_feature_embed_fails_LOUD(self, ckpt):
        """``feature_vocab_from_state_dict`` only reads ``feature_embed``, so a
        ``value_embeds`` table of the wrong width is NOT caught by the #695 guards. It is
        caught by ``load_state_dict(strict=True)`` — which shape-checks even under
        ``assign=True``.

        That is the whole point of pinning it: ``assign=True`` is the flag people reach for
        precisely to skip copies, and if a future PyTorch (or a switch to ``strict=False``)
        ever let it through, every value-embedding lookup would silently read the wrong
        rows. ``FeatureVocabError`` is NOT a ``RuntimeError``, so this cannot pass by
        accidentally catching one of our own guards.
        """
        m = _build(VOCAB)
        _save(ckpt, m)
        ve_key = next(k for k in m.state_dict() if "value_embeds" in k and k.endswith("weight"))
        d = _tamper_sd(
            ckpt,
            lambda sd: sd.__setitem__(ve_key, t.zeros(N_SAE, sd[ve_key].shape[1])),
        )
        with pytest.raises(RuntimeError, match="size mismatch"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )

    def test_a_meta_mirror_lying_about_vocab_size_raises(self, ckpt):
        """The mirror is a 3-field dict. Tampering ONLY ``sha256`` (the existing test) would
        still pass if the loader compared just that one field."""
        _save(ckpt, _build(VOCAB))
        d = _tamper_meta(ckpt, lambda meta: meta["feature_vocab"].__setitem__("vocab_size", 99))
        with pytest.raises(FeatureVocabError, match="disagrees with the weights"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )

    def test_a_meta_mirror_lying_about_n_sae_features_raises(self, ckpt):
        _save(ckpt, _build(VOCAB))
        d = _tamper_meta(
            ckpt, lambda meta: meta["feature_vocab"].__setitem__("n_sae_features", 128)
        )
        with pytest.raises(FeatureVocabError, match="disagrees with the weights"):
            build_encdec_model(
                d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
            )


class TestTheRemapIsCorrectAfterLoad:
    """The compact-id remap is derived per-forward from the PERSISTENT ``feature_vocab_ids``
    buffer, which ``load_state_dict`` restores. There is no separately-stored map to go stale,
    so the failure modes reduce to: (a) the ids buffer disagrees with the configured vocab
    (``assert_vocab_consistent`` catches it), and (b) an out-of-vocab id at forward time (the
    ``_to_compact`` assert catches it). Both are proven to still bite AFTER a round trip.
    """

    # Deleted with the derived buffer: the ZEROED-map / stale-map tests. A zeroed non-persistent
    # map that silently collapsed every feature onto row 0 was the whole hazard the old machinery
    # guarded; with the map computed on the fly from feature_vocab_ids, that state cannot exist.

    def test_assert_vocab_consistent_CATCHES_a_tampered_ids_buffer(self, ckpt):
        d = _save(ckpt, _build(VOCAB))
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        loaded.encoder.feature_vocab_ids[0] = 1  # 0 -> 1: still sorted, still in range
        with pytest.raises(FeatureVocabError, match="feature_vocab_ids disagrees with the model"):
            loaded.assert_vocab_consistent()

    def test_assert_vocab_consistent_CATCHES_a_tampered_n_sae_buffer(self, ckpt):
        d = _save(ckpt, _build(VOCAB))
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        loaded.encoder.feature_vocab_n_sae.fill_(N_SAE * 2)
        with pytest.raises(FeatureVocabError, match=r"feature_vocab_n_sae=\d+ disagrees"):
            loaded.assert_vocab_consistent()

    def test_the_loaded_model_still_REJECTS_an_out_of_vocab_feature_id(self, ckpt):
        """The OOV assert derives from ``feature_vocab_ids`` via searchsorted. Prove it still
        bites AFTER a round trip, not just on a freshly-constructed model."""
        d = _save(ckpt, _build(VOCAB))
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        assert 5 not in ALLOWED
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            _fwd(loaded, feature_ids=(10, 5, 30))

    def test_and_still_ACCEPTS_every_id_that_IS_in_the_vocabulary(self, ckpt):
        """The other half — a guard that rejects everything would pass the test above."""
        d = _save(ckpt, _build(VOCAB))
        loaded, _, _ = build_encdec_model(
            d, 5, "cpu", ModelPhase.TRAIN, eval_conditioning=EvalConditioning.not_evaluating()
        )
        for orig in ALLOWED:
            assert t.isfinite(_fwd(loaded, feature_ids=(orig, orig, orig))).all()


class TestResumeIsSTRICTLyMaskEqual:
    """A checkpoint compacted with mask A must be REFUSED by a run configured with mask B.

    Strict EQUALITY of the allowed universe — not subset, not superset. Anything else
    silently reinterprets every embedding row, and for a same-size mask the shapes match
    so ``load_state_dict`` would happily succeed.

    Each test drives ``_resume_into``, i.e. the real guard sequence from
    ``conditional_encdec_train.py``, against a real checkpoint on disk.
    """

    def test_same_mask_resume_loads_and_is_BITWISE(self, ckpt):
        trained = _build(VOCAB, seed=3)
        expected = _fwd(trained)
        d = _save(ckpt, trained)

        fresh = _build(VOCAB, seed=99)  # different weights entirely
        assert not t.equal(_fwd(fresh), expected), "the fixture is toothless otherwise"

        _resume_into(fresh, d)
        assert t.equal(_fwd(fresh), expected)

    def test_a_DIFFERENT_mask_of_the_SAME_SIZE_is_refused(self, ckpt):
        """The dangerous one: V matches, so every tensor shape matches, so
        ``load_state_dict`` would SUCCEED and hand you a silently wrong dictionary."""
        d = _save(ckpt, _build(VOCAB))
        other = FeatureVocab.from_mask(MASK_SAME_SIZE, N_SAE)
        assert other is not None and other.size == VOCAB.size, "shapes must collide"
        with pytest.raises(FeatureVocabError, match="feature vocabulary mismatch"):
            _resume_into(_build(other), d)

    def test_a_STRICT_SUBSET_mask_is_refused(self, ckpt):
        """Not 'compatible enough'. A subset would need every row re-indexed."""
        d = _save(ckpt, _build(VOCAB))
        sub = FeatureVocab.from_mask(MASK_SUBSET, N_SAE)
        assert sub is not None and set(sub.original_ids) < set(VOCAB.original_ids)
        with pytest.raises(FeatureVocabError):
            _resume_into(_build(sub), d)

    def test_a_STRICT_SUPERSET_mask_is_refused(self, ckpt):
        d = _save(ckpt, _build(VOCAB))
        sup = FeatureVocab.from_mask(MASK_SUPERSET, N_SAE)
        assert sup is not None and set(sup.original_ids) > set(VOCAB.original_ids)
        with pytest.raises(FeatureVocabError):
            _resume_into(_build(sup), d)

    def test_the_refusal_comes_from_OUR_guard_BEFORE_load_state_dict(self, ckpt):
        """A subset/superset also trips PyTorch's shape check — but only AFTER the model
        has been half-mutated, and with a message that names no remedy. Pin that our guard
        wins the race, or the actionable error is dead code."""
        d = _save(ckpt, _build(VOCAB))
        sub = FeatureVocab.from_mask(MASK_SUBSET, N_SAE)
        assert sub is not None
        model = _build(sub)
        model_data, _, _ = load_checkpoint(d, 5, "cpu", load_optimizer=False, rank=0)
        with pytest.raises(FeatureVocabError) as exc:
            assert_checkpoint_vocab_compatible(model.feature_vocab, model_data, checkpoint_dir=d)
        assert "size mismatch" not in str(exc.value), "PyTorch's message must not win"
        assert d in str(exc.value), "the operator must be told WHICH checkpoint"

    def test_the_REFUSED_model_is_left_COMPLETELY_UNMUTATED(self, ckpt):
        """'Before load_state_dict' stated behaviourally rather than by call-order.

        A same-size wrong mask passes every shape check, so if the guard ran even one
        step late, ``load_state_dict`` would have already overwritten the weights by the
        time it fired. Then a caller that caught ``FeatureVocabError`` and carried on —
        or an operator reading a traceback and assuming nothing happened — would be
        holding a half-swapped model. Assert the model is byte-for-byte what it was.
        """
        d = _save(ckpt, _build(VOCAB))
        other = FeatureVocab.from_mask(MASK_SAME_SIZE, N_SAE)
        assert other is not None and other.size == VOCAB.size
        model = _build(other)
        before = {k: v.clone() for k, v in model.state_dict().items()}

        with pytest.raises(FeatureVocabError):
            _resume_into(model, d)

        after = model.state_dict()
        assert set(before) == set(after)
        for key, tensor in before.items():
            assert t.equal(tensor, after[key]), (
                f"{key} was mutated before the guard fired"
            )

    def test_an_UNMASKED_run_refuses_a_compacted_checkpoint(self, ckpt):
        d = _save(ckpt, _build(VOCAB))
        with pytest.raises(FeatureVocabError, match="dump_feature_mask"):
            _resume_into(_build(None), d)

    def test_a_MASKED_run_refuses_a_LEGACY_uncompacted_checkpoint(self, ckpt):
        d = _save(ckpt, _build(None))
        with pytest.raises(FeatureVocabError, match="feature_mask_freq_min"):
            _resume_into(_build(VOCAB), d)

    def test_a_BLACKLIST_resolving_to_the_SAME_universe_resumes_and_is_BITWISE(self, ckpt):
        """The vocab keys on the allowed UNIVERSE, not on how the mask was spelled.

        A blacklist of the complement and a whitelist of ALLOWED are the same dictionary,
        so the encoder's rows mean the same thing and the resume must succeed. (The mask
        SPELLING is separately guarded by ``assert_resume_feature_mask_matches``; conflating
        the two here would make the vocab guard reject a legitimate re-spelling.)
        """
        from_wl = FeatureVocab.from_mask(MASK_A, N_SAE)
        from_bl = FeatureVocab.from_mask(MASK_A_AS_BLACKLIST, N_SAE)
        assert from_wl == from_bl == VOCAB
        assert MASK_A.dirname() != MASK_A_AS_BLACKLIST.dirname(), "different masks..."

        trained = _build(from_wl, seed=3)
        expected = _fwd(trained)
        d = _save(ckpt, trained)

        fresh = _build(from_bl, seed=99)
        _resume_into(fresh, d)
        assert t.equal(_fwd(fresh), expected)

    def test_same_ids_but_a_DIFFERENT_n_sae_is_refused(self, ckpt):
        """The ``bl_``/``wl_`` checkpoint leaf hashes (mode, ids) and NOT n_sae, so a 16k
        and a 65k SAE with the same whitelist land in the SAME directory with the same V.
        Only this check stands between them."""
        d = _save(ckpt, _build(VOCAB))
        wider = FeatureVocab(n_sae_features=N_SAE * 2, original_ids=ALLOWED)
        model = _build(VOCAB)  # correct shapes...
        model.encoder.config.n_sae_features = N_SAE * 2  # ...but a different id space
        model.encoder.feature_vocab = wider
        model_data, _, _ = load_checkpoint(d, 5, "cpu", load_optimizer=False, rank=0)
        with pytest.raises(FeatureVocabError, match="n_sae_features mismatch"):
            assert_checkpoint_vocab_compatible(wider, model_data, checkpoint_dir=d)
