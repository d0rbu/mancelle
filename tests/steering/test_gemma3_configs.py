"""Schema/sanity tests for the new Gemma-3 config siblings."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from hydra import compose, initialize_config_dir

import steering.eval.axbench_bpb.runtime_state as runtime_state_module
from scripts.eval.axbench_bpb_eval import _cfg_to_dataclass
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.sae_resolver import ResolvedSAE

GEMMA3_EXTRACTION_CONFIGS = [
    "configs/extraction/extract_features_gemma3.yaml",
]

GEMMA3_DOWNSTREAM_CONFIGS = [
    "configs/extraction/compute_statistics_gemma3.yaml",
    "configs/extraction/precompute_percentiles_gemma3.yaml",
    "configs/extraction/precompute_token_boundaries_gemma3.yaml",
    "configs/extraction/precompute_packed_rows_gemma3.yaml",
    "configs/extraction/validate_token_boundary_overlays_gemma3.yaml",
]


@pytest.mark.parametrize("path", GEMMA3_EXTRACTION_CONFIGS)
def test_config_file_exists(path):
    assert Path(path).exists(), f"missing config: {path}"


@pytest.mark.parametrize("path", GEMMA3_EXTRACTION_CONFIGS)
def test_config_is_valid_yaml(path):
    if not Path(path).exists():
        pytest.skip(f"{path} not yet created")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)


class TestGemma3ExtractConfigComposes:
    def test_compose_resolves_gemma3_v2_defaults(self):
        cfg_dir = str(Path("configs/extraction").resolve())
        with initialize_config_dir(version_base=None, config_dir=cfg_dir):
            cfg = compose(config_name="extract_features_gemma3")
        assert cfg.model == "gemma-3-4b-it"
        assert cfg.sae.release == "gemmascope_2"
        assert str(cfg.sae.l0) == "medium"
        assert str(cfg.sae.num_features) == "16k"
        assert cfg.layer == 22

    def test_inherited_fields_come_through_from_parent(self):
        cfg_dir = str(Path("configs/extraction").resolve())
        with initialize_config_dir(version_base=None, config_dir=cfg_dir):
            cfg = compose(config_name="extract_features_gemma3")
        # These are NOT redeclared in the v2 config; they must inherit
        # from extract_features.yaml.
        assert cfg.segment_size == 128
        assert cfg.context_window == 1024
        assert cfg.pooling == "max_pool"
        assert cfg.activation_type == "residual"


GEMMA3_TRAINING_EVAL_CONFIGS = [
    "configs/training/conditional_encdec_train_gemma3.yaml",
    "configs/eval/base_eval_gemma3.yaml",
    "configs/eval/chat_eval_gemma3.yaml",
    "configs/eval/completion_steering_eval_gemma3.yaml",
    "configs/eval/axbench_bpb/axbench_bpb_eval_gemma3.yaml",
]


@pytest.mark.parametrize("path", GEMMA3_TRAINING_EVAL_CONFIGS)
def test_training_eval_config_exists(path):
    assert Path(path).exists(), f"missing: {path}"


class TestGemma3TrainingEvalCompose:
    """Compose / structural sanity for the training + eval gemma3 siblings.

    None of these 5 parents expose a top-level ``model:`` key (unlike the
    extraction configs): the model identity is nested under ``sae:`` (training),
    ``conditioning.identity`` (base/chat eval), or ``neuronpedia`` /
    flat ``neuronpedia_model_id`` (completion / axbench). Assertions target the
    actual nested coordinate per parent.

    base_eval / chat_eval / axbench_bpb_eval declare mandatory ``???`` fields
    (source, val_packed_row_dir, cache_dir, source_run_dir, ...). OmegaConf does
    not error on missing-mandatory at *compose* time — only on attribute access
    — so ``compose`` succeeds and we only read the overridden gemma3 keys.
    """

    def test_train_config_composes(self):
        with initialize_config_dir(
            version_base=None,
            config_dir=str(Path("configs/training").resolve()),
        ):
            cfg = compose(config_name="conditional_encdec_train_gemma3")
        # Encdec training has no top-level model: the feature-source model
        # lives in the structured sae: block (flat width, not v2 num_features).
        assert cfg.sae.model == "gemma-3-4b-it"
        assert cfg.sae.release == "gemmascope_2"
        assert cfg.sae.layer == 22
        assert str(cfg.sae.width) == "16k"
        assert cfg.sae.activation_type == "residual"
        # Inherited (NOT redeclared) field from conditional_encdec_train.yaml.
        assert cfg.segment_size == 128

    @pytest.mark.parametrize("name", ["base_eval_gemma3", "chat_eval_gemma3"])
    def test_eval_conditioning_identity_composes(self, name):
        with initialize_config_dir(
            version_base=None,
            config_dir=str(Path("configs/eval").resolve()),
        ):
            cfg = compose(config_name=name)
        assert cfg.conditioning.identity.model_id == "gemma-3-4b-it"
        assert cfg.conditioning.identity.layer == 22
        assert "l22" in str(cfg.conditioning.identity.sae_id)
        # Inherited (NOT redeclared): n_sae_features stays 16384.
        assert cfg.conditioning.n_sae_features == 16384

    def test_completion_steering_neuronpedia_composes(self):
        with initialize_config_dir(
            version_base=None,
            config_dir=str(Path("configs/eval").resolve()),
        ):
            cfg = compose(config_name="completion_steering_eval_gemma3")
        assert cfg.neuronpedia.model_id == "gemma-3-4b-it"
        assert cfg.neuronpedia.sae_source == "22-gemmascope-2-res-16k"
        assert "gemmascope-2" in cfg.neuronpedia.sae_source
        # Inherited (NOT redeclared) field from the v1 parent.
        assert cfg.segment_size == 128

    def test_axbench_bpb_gemma3_neuronpedia_source(self):
        # axbench_bpb_eval.yaml has a defaults: chain only to its same-dir
        # sibling, but a raw structural check is the spec-mandated minimum.
        with open(
            "configs/eval/axbench_bpb/axbench_bpb_eval_gemma3.yaml"
        ) as f:
            raw = yaml.safe_load(f)
        assert raw["extraction_tokenizer_id"] == "google/gemma-3-4b-it"
        assert raw["neuronpedia_model_id"] == "gemma-3-4b-it"
        src = str(raw.get("neuronpedia_source", ""))
        # The v2 Neuronpedia source uses the gemmascope-2 grammar.
        assert "gemmascope-2" in src, src
        assert src == "22-gemmascope-2-res-16k", src

    def test_axbench_bpb_gemma3_composes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        with initialize_config_dir(
            version_base=None,
            config_dir=str(Path("configs/eval/axbench_bpb").resolve()),
        ):
            cfg = compose(
                config_name="axbench_bpb_eval_gemma3",
                overrides=[
                    "source_run_dir=/tmp/source",
                    "output_root=/tmp/output",
                    "feature_dir=/tmp/features",
                    "feature_statistics_path=/tmp/statistics.safetensors",
                    "decoder_only.source=local",
                    "decoder_only.model_tag=test",
                    "decoder_only.step=1",
                    "encdec.source=local",
                    "encdec.model_tag=test",
                    "encdec.step=1",
                    "device=cpu",
                ],
            )
        assert cfg.extraction_tokenizer_id == "google/gemma-3-4b-it"
        assert cfg.neuronpedia_model_id == "gemma-3-4b-it"
        assert cfg.neuronpedia_source == "22-gemmascope-2-res-16k"
        assert cfg.diffmean_and_probe_layer == 22
        # Inherited (NOT redeclared) field from the v1 parent.
        assert cfg.context_cache.segment_size == 256

        captured: dict[str, object] = {}

        def _load_sae_and_model(**kwargs: object):
            captured.update(kwargs)
            return (
                object(),
                SimpleNamespace(cfg=SimpleNamespace(d_sae=16384)),
                object(),
            )

        monkeypatch.setattr(
            runtime_state_module,
            "load_sae_and_model",
            _load_sae_and_model,
        )
        state = RuntimeState(
            cfg=_cfg_to_dataclass(cfg),
            ddp_rank=0,
            ddp_world_size=1,
        )

        state.base_gemma_and_sae()

        resolved = captured["resolved"]
        assert isinstance(resolved, ResolvedSAE)
        assert resolved == ResolvedSAE(
            release="gemma-scope-2-4b-it-res",
            sae_id="layer_22_width_16k_l0_medium",
            hook_name="blocks.22.hook_resid_post",
        )


@pytest.mark.parametrize("path", GEMMA3_DOWNSTREAM_CONFIGS)
def test_downstream_config_exists(path):
    assert Path(path).exists(), f"missing: {path}"


class TestGemma3DownstreamConfigsCompose:
    @pytest.mark.parametrize(
        "config_name",
        [
            "compute_statistics_gemma3",
            "precompute_percentiles_gemma3",
            "precompute_token_boundaries_gemma3",
            "precompute_packed_rows_gemma3",
            "validate_token_boundary_overlays_gemma3",
        ],
    )
    def test_compose_resolves_gemma3_model(self, config_name):
        cfg_dir = str(Path("configs/extraction").resolve())
        with initialize_config_dir(version_base=None, config_dir=cfg_dir):
            cfg = compose(config_name=config_name)
        # Every downstream gemma3 sibling must target a Gemma 3 model.
        model_val = cfg.get("model") if hasattr(cfg, "get") else cfg.model
        assert str(model_val).startswith("gemma-3"), f"{config_name}: model={model_val}"

    @pytest.mark.parametrize(
        "config_name",
        [
            "compute_statistics_gemma3",
            "precompute_percentiles_gemma3",
            "precompute_token_boundaries_gemma3",
            "precompute_packed_rows_gemma3",
            "validate_token_boundary_overlays_gemma3",
        ],
    )
    def test_inherited_fields_and_layer_override(self, config_name):
        cfg_dir = str(Path("configs/extraction").resolve())
        with initialize_config_dir(version_base=None, config_dir=cfg_dir):
            cfg = compose(config_name=config_name)
        # Layer overridden to a valid gemma-scope-2 4b site (matches
        # extract_features_gemma3.yaml).
        assert cfg.layer == 22
        # These are NOT redeclared in the gemma3 sibling; they must inherit
        # from the v1 parent.
        assert cfg.pooling == "max_pool"
        assert cfg.activation_type == "residual"
        assert cfg.segment_size == 128
        assert str(cfg.sae_width) == "16k"
