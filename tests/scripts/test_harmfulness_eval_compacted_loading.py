"""Integration coverage for compacted harmfulness-eval checkpoint loading."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch as t
from omegaconf import OmegaConf

from nanochat.checkpoint_manager import ModelPhase
from scripts.eval import harmfulness_eval
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval import dclm_eval
from steering.refined_types import PositiveInt

N_SAE = PositiveInt.parse(64)
NUM_BUCKETS = PositiveInt.parse(2)
STEP = 16
CHECKPOINT_DIR = "/checkpoints/capabilities_sft_conditioned_checkpoints"
MODEL_IDENTITY = {
    "format": "nanochat_checkpoint",
    "backend": "nanochat",
    "architecture": "EncoderDecoderGPT",
    "model_type": "encoder_decoder",
    "step": STEP,
    "checkpoint_sha256": "1" * 64,
    "tokenizer_sha256": "2" * 64,
}


def _mask(ids: tuple[int, ...] = tuple(range(8, 24))) -> FeatureMask:
    return FeatureMask(FeatureMaskMode.whitelist, ids)


def _clusters() -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(
        clusters=((8, 9, 10), (12, 13), (20, 21)),
    )
    return ResolvedFeatureClusters(
        source_path=Path("cluster-source.json"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=N_SAE,
        ),
    )


def _checkpoint_meta(
    *,
    clusters: ResolvedFeatureClusters | None,
) -> tuple[dict, FeatureVocab]:
    mask = _mask()
    universe = mask if clusters is None else FeatureClusterConsolidation.build(clusters, mask)
    feature_vocab = effective_feature_vocab(universe, N_SAE)
    assert feature_vocab is not None
    meta = {
        "feature_mask_canonical": mask.canonical_dict(),
        **feature_clusters_checkpoint_metadata(clusters),
        "user_config": {
            "num_buckets": NUM_BUCKETS,
            "negative_selection": {"distribution": "normal"},
        },
    }
    return meta, feature_vocab


class _EncDecStub(EncoderDecoderGPT):
    """Production-type stub carrying only the load/selection contract state."""

    def __init__(self, feature_vocab: FeatureVocab) -> None:
        t.nn.Module.__init__(self)
        self.encoder = SimpleNamespace(
            config=SimpleNamespace(
                n_sae_features=N_SAE,
                num_buckets=NUM_BUCKETS,
            ),
            feature_vocab=feature_vocab,
        )
        self.dec_config = SimpleNamespace(sequence_len=16)
        self.consistency_calls = 0

    def assert_vocab_consistent(self) -> None:
        self.consistency_calls += 1


def _load_cfg(source: str):
    return OmegaConf.create(
        {
            "source": source,
            "model_tag": "d16",
            "step": STEP,
            "segment_size": None,
            "feature_selection": None,
            "feature_mask": None,
        }
    )


def _mock_artifact_identity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutation_error: ValueError | None = None,
) -> SimpleNamespace:
    artifacts = SimpleNamespace(
        checkpoint_dir=Path(CHECKPOINT_DIR).resolve(),
        step=STEP,
        checkpoint_inventory=SimpleNamespace(entries=[SimpleNamespace(path=Path("model.pt"), sha256="a" * 64)]),
        assert_unchanged=MagicMock(side_effect=mutation_error),
    )
    monkeypatch.setattr(
        harmfulness_eval,
        "capture_native_artifact_identity",
        MagicMock(return_value=artifacts),
    )
    monkeypatch.setattr(
        harmfulness_eval,
        "native_model_identity",
        MagicMock(return_value=MODEL_IDENTITY),
    )
    return artifacts


def test_direct_encdec_branch_loads_compacted_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta, feature_vocab = _checkpoint_meta(clusters=None)
    model = _EncDecStub(feature_vocab)

    def build_stub(
        checkpoint_dir: str,
        step: int,
        device: t.device,
        phase: ModelPhase,
        *,
        eval_conditioning,
    ):
        assert (checkpoint_dir, step, device, phase) == (
            CHECKPOINT_DIR,
            STEP,
            t.device("cpu"),
            ModelPhase.EVAL,
        )
        eval_conditioning.assert_loadable(
            feature_vocab,
            checkpoint_dir=checkpoint_dir,
        )
        return model, object(), dict(meta)

    monkeypatch.setattr(
        harmfulness_eval,
        "_resolve_checkpoint_dir_for_harmfulness",
        lambda **_kwargs: CHECKPOINT_DIR,
    )
    monkeypatch.setattr(harmfulness_eval, "build_encdec_model", build_stub)
    artifacts = _mock_artifact_identity(monkeypatch)

    loaded_model, _tokenizer, loaded_meta, loaded_contract, loaded_identity = (
        harmfulness_eval._load_model_for_harmfulness(
            _load_cfg("encdec"),
            t.device("cpu"),
        )
    )

    assert loaded_model is model
    assert loaded_meta["checkpoint_dir"] == CHECKPOINT_DIR
    assert loaded_contract == CheckpointFeatureContract(_mask(), None)
    assert loaded_identity == MODEL_IDENTITY
    artifacts.assert_unchanged.assert_called_once_with()
    assert model.consistency_calls == 1


def test_direct_encdec_branch_accepts_turn_checkpoint_for_authenticated_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta, feature_vocab = _checkpoint_meta(clusters=None)
    meta["sft_segment_mode"] = "turn_delimited_fixed_tokens"
    model = _EncDecStub(feature_vocab)
    monkeypatch.setattr(
        harmfulness_eval,
        "_resolve_checkpoint_dir_for_harmfulness",
        lambda **_kwargs: CHECKPOINT_DIR,
    )
    monkeypatch.setattr(
        harmfulness_eval,
        "build_encdec_model",
        lambda *_args, **_kwargs: (model, object(), dict(meta)),
    )
    artifacts = _mock_artifact_identity(monkeypatch)

    loaded, _, loaded_meta, _, _ = harmfulness_eval._load_model_for_harmfulness(
        _load_cfg("encdec"), t.device("cpu"))
    assert loaded is model
    assert loaded_meta["sft_segment_mode"] == "turn_delimited_fixed_tokens"
    artifacts.assert_unchanged.assert_called_once()
    assert isinstance(harmfulness_eval.native_model_identity, MagicMock)
    harmfulness_eval.native_model_identity.assert_called_once()


def test_autodetected_branch_loads_clustered_compacted_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta, feature_vocab = _checkpoint_meta(clusters=_clusters())
    meta["checkpoint_dir"] = CHECKPOINT_DIR
    model = _EncDecStub(feature_vocab)

    def load_stub(**kwargs):
        assert kwargs["source"] == "capabilities_sft_conditioned_checkpoints"
        assert kwargs["phase"] is ModelPhase.EVAL
        return model, object(), dict(meta)

    monkeypatch.setattr(harmfulness_eval, "load_model", load_stub)
    monkeypatch.setattr(
        harmfulness_eval,
        "_resolve_checkpoint_dir_for_harmfulness",
        lambda **_kwargs: CHECKPOINT_DIR,
    )
    artifacts = _mock_artifact_identity(monkeypatch)

    loaded_model, _tokenizer, loaded_meta, loaded_contract, loaded_identity = (
        harmfulness_eval._load_model_for_harmfulness(
            _load_cfg("capabilities_sft_conditioned_checkpoints"),
            t.device("cpu"),
        )
    )

    assert loaded_model is model
    assert loaded_meta == {**meta, "resolved_step": STEP, "loaded_model_sha256": "a" * 64}
    assert loaded_contract.feature_mask == _mask()
    assert loaded_contract.resolved_clusters is not None
    assert loaded_contract.resolved_clusters.canonical_dict() == _clusters().canonical_dict()
    assert model.consistency_calls == 1
    assert loaded_identity == MODEL_IDENTITY
    artifacts.assert_unchanged.assert_called_once_with()


def test_latest_step_is_pinned_and_checkpoint_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _load_cfg("encdec")
    cfg.step = None
    model = MagicMock()
    find_step = MagicMock(return_value=STEP)
    build = MagicMock(return_value=(model, object(), {}))
    monkeypatch.setattr(
        harmfulness_eval,
        "_resolve_checkpoint_dir_for_harmfulness",
        lambda **_kwargs: CHECKPOINT_DIR,
    )
    monkeypatch.setattr(harmfulness_eval, "find_last_step", find_step)
    monkeypatch.setattr(harmfulness_eval, "build_encdec_model", build)
    monkeypatch.setattr(
        harmfulness_eval,
        "assert_conditioned_eval_model",
        lambda *_args, **_kwargs: object(),
    )
    artifacts = _mock_artifact_identity(
        monkeypatch,
        mutation_error=ValueError("checkpoint changed while the model was loading"),
    )

    with pytest.raises(ValueError, match="changed while the model was loading"):
        harmfulness_eval._load_model_for_harmfulness(cfg, t.device("cpu"))

    find_step.assert_called_once_with(CHECKPOINT_DIR)
    assert build.call_args.args[1] == STEP
    artifacts.assert_unchanged.assert_called_once_with()


def test_signed_target_outside_compacted_vocabulary_fails_before_generation() -> None:
    with pytest.raises(ValueError, match="hand-supplied feature_configs"):
        harmfulness_eval._validate_harmfulness_generation_config(
            feature_configs_configured=True,
            self_cond_segment_size=None, is_encdec=True, batch_size=1)


def test_dclm_eval_retains_cluster_rejection_guard() -> None:
    source = inspect.getsource(dclm_eval)

    assert 'reject_feature_clusters_metadata(meta, "dclm_eval checkpoint")' in source


@pytest.mark.parametrize("source", ["encdec", "capabilities_sft_conditioned_checkpoints"])
def test_cuda_encdec_load_stages_on_cpu_then_transfers_once(monkeypatch, tmp_path, source):
    meta, vocab = _checkpoint_meta(clusters=None)
    meta.update(checkpoint_dir=CHECKPOINT_DIR, encoder_config={"fixture": True}, xattn_config={"fixture": True})
    model = _EncDecStub(vocab)
    transfer = MagicMock(return_value=model)
    monkeypatch.setattr(model, "to", transfer)
    devices = []

    def build(_directory, _step, device, _phase, **kwargs):
        devices.append(device)
        return model, object(), dict(meta)

    def load(**kwargs):
        devices.append(kwargs["device"])
        return model, object(), dict(meta)

    monkeypatch.setattr(harmfulness_eval, "build_encdec_model", build)
    monkeypatch.setattr(harmfulness_eval, "load_model", load)
    monkeypatch.setattr(harmfulness_eval, "_resolve_checkpoint_dir_for_harmfulness", lambda **kwargs: CHECKPOINT_DIR)
    artifacts = _mock_artifact_identity(monkeypatch)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(meta))
    artifacts.checkpoint_inventory.entries.append(SimpleNamespace(path=metadata_path, sha256="b" * 64))
    loaded, *_ = harmfulness_eval._load_model_for_harmfulness(_load_cfg(source), t.device("cuda"))
    assert loaded is model
    assert devices == [t.device("cpu")]
    transfer.assert_called_once_with(device=t.device("cuda"), dtype=t.bfloat16)
