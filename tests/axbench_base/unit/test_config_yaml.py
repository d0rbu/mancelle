"""Tests for the ``configs/eval/axbench_base`` Hydra config tree.

Covers:
 - the root YAML composes via ``hydra.compose`` with every tier/stage/
   methods defaults combination,
 - every mandatory ``AxBenchBaseConfig`` field is referenced (either
   explicit in the root YAML or required at convert-time),
 - each tier file sets ``max_concepts`` to its specified value,
 - ``cfg_to_dataclass`` converts a composed OmegaConf dict to an
   ``AxBenchBaseConfig`` with nested ``JudgeConfig``, ``SynthConfig``,
   ``GenerationConfig``, ``LoraConfig``, ``LayerHooksConfig``, and
   ``LLMClientConfig`` (concept_examples_qc_judge) instances,
 - unknown keys fail when struct mode is enabled,
 - missing mandatory fields raise ``AxBenchBaseError``.

The tests use the project's ``configs/eval/axbench_base`` directory
resolved relative to the repo root (the worktree root), so they work
regardless of the pytest invocation CWD.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

import scripts.eval.axbench_base_eval as axbench_cli
from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    GenerationConfig,
    JudgeConfig,
    LayerHooksConfig,
    LLMClientConfig,
    LoraConfig,
    ProfileConfig,
    SynthConfig,
    validate_config,
)
from steering.eval.axbench_base.config_loader import cfg_to_dataclass
from steering.eval.axbench_base.types import AxBenchBaseError

_CONFIG_DIR = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "eval"
    / "axbench_base"
)

_MANDATORY_OVERRIDES: list[str] = [
    "feature_dir=out/data/features/test",
    "feature_statistics_path=out/data/features/test/stats.safetensors",
    "decoder_only.source=base",
    "decoder_only.model_tag=d13",
    "decoder_only.step=10000",
    "encdec.source=encdec",
    "encdec.model_tag=d13",
    "encdec.step=10000",
]


def _compose_root(
    tier: str = "standard",
    stages: str = "all",
    methods: str = "all",
    extra: list[str] | None = None,
    with_mandatory: bool = True,
) -> DictConfig:
    """Compose the root axbench_base config with the given overrides."""
    overrides: list[str] = [f"tiers={tier}", f"stages={stages}", f"methods={methods}"]
    if with_mandatory:
        overrides.extend(_MANDATORY_OVERRIDES)
    if extra:
        overrides.extend(extra)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="axbench_base", overrides=overrides)
    return cfg


# ---------------------------------------------------------------------------
# File presence
# ---------------------------------------------------------------------------


class TestConfigFilesExist:
    """The root + subconfig files must all live on disk."""

    def test_root_yaml_exists(self) -> None:
        assert (_CONFIG_DIR / "axbench_base.yaml").is_file()

    @pytest.mark.parametrize("tier", ["dev", "standard", "full"])
    def test_tier_file_exists(self, tier: str) -> None:
        assert (_CONFIG_DIR / "tiers" / f"{tier}.yaml").is_file()

    def test_stages_all_file_exists(self) -> None:
        assert (_CONFIG_DIR / "stages" / "all.yaml").is_file()

    def test_methods_all_file_exists(self) -> None:
        assert (_CONFIG_DIR / "methods" / "all.yaml").is_file()


# ---------------------------------------------------------------------------
# Root YAML loads as raw OmegaConf
# ---------------------------------------------------------------------------


class TestRootConfigRaw:
    """Raw ``OmegaConf.load`` of the root YAML (no Hydra composition)."""

    def test_root_config_loads_via_omegaconf(self) -> None:
        """OmegaConf.load returns a DictConfig with the expected
        top-level keys (defaults, feature_dir, etc.)."""
        cfg = OmegaConf.load(_CONFIG_DIR / "axbench_base.yaml")
        assert isinstance(cfg, DictConfig)
        # Every mandatory field must appear — either as an explicit key
        # (set to ??? or a value) or through a nested group. Here we
        # assert the six mandatory top-level keys are present. NB: we
        # inspect ``.keys()`` rather than using ``in`` because a key
        # with a ``???`` MISSING value is not a member of the DictConfig
        # via membership test.
        present_keys = set(cfg.keys())
        for key in (
            "feature_dir",
            "feature_statistics_path",
            "decoder_only",
            "encdec",
            "device",
            "output_root",
        ):
            assert key in present_keys, (
                f"root YAML must reference mandatory field {key!r}"
            )
        assert "conditioning" in present_keys, (
            "root YAML must expose the Stage 4 encoder-decoder full-context "
            "cache configuration block"
        )
        assert cfg.verification_prefill_mode == "shared"
        assert "batched_generation_mode" not in present_keys

    def test_root_defaults_list_includes_tier_stage_method(self) -> None:
        """Parse the raw file text to confirm the ``defaults`` list
        contains ``tiers``, ``stages``, ``methods``, ``_self_``."""
        text = (_CONFIG_DIR / "axbench_base.yaml").read_text()
        # The defaults list is a YAML list, so grep for the bullet lines.
        assert "defaults:" in text
        assert "- tiers: standard" in text
        assert "- stages: all" in text
        assert "- methods: all" in text
        assert "- _self_" in text


# ---------------------------------------------------------------------------
# Hydra compose
# ---------------------------------------------------------------------------


class TestHydraComposeAPI:
    """Use ``hydra.compose`` to build the full config with overrides."""

    def test_hydra_compose_api_works(self) -> None:
        cfg = _compose_root()
        assert cfg is not None
        assert cfg.max_concepts == 50  # standard tier
        assert cfg.stage == "all"
        assert cfg.enabled_methods is None
        assert cfg.conditioning.context_cache_dir is None
        assert cfg.conditioning.n_sae_features == 16384
        assert cfg.conditioning.num_buckets == 2

    def test_gemma3_4bpt_l22_f65k_config_uses_matching_neuronpedia_source(
        self,
    ) -> None:
        """The gemma-3-4b-pt l22 f65k AxBench-Base config points Stage 0a at
        the matching 65k Neuronpedia source (issue #657), not the gemma-2-2b
        16k default that would 404 every index >= 16384 and trip the Stage 0a
        source-mismatch guard. Composes to a valid dataclass."""
        GlobalHydra.instance().clear()
        # feature_dir must be a canonical gemma-3 path: validate_config parses
        # the SAE model from it and checks it against the extraction model
        # (issue #657), so the generic out/data/features/test placeholder in
        # _MANDATORY_OVERRIDES would (correctly) fail the canonical-path check.
        gemma3_overrides = [
            o for o in _MANDATORY_OVERRIDES if not o.startswith("feature_dir=")
        ] + [
            "feature_dir=out/data/features/s128/l22/a_res/p_max_pool/f65k/"
            "m_gemma-3-4b-pt/base",
        ]
        with initialize_config_dir(
            config_dir=str(_CONFIG_DIR), version_base=None,
        ):
            cfg = compose(
                config_name="axbench_base_gemma3_4bpt_l22_f65k",
                overrides=gemma3_overrides,
            )
        assert cfg.neuronpedia_model_id == "gemma-3-4b"
        assert cfg.neuronpedia_source == "22-gemmascope-2-res-65k"
        assert cfg.feature_extraction_model_id == "gemma-3-4b-pt"
        assert cfg.extraction_tokenizer_id == "google/gemma-3-4b-pt"
        # NOT 22. This field indexes the model the capture hook runs on -- the
        # nanochat encdec decoder (12 blocks) -- while 22 is the gemma-3-4b-pt
        # SAE site asserted two lines above. This assertion previously pinned
        # 22 and so locked in the bug: stage 3a died on `IndexError: index 22
        # is out of range`. The two model identities must stay decoupled here.
        assert cfg.layer_hooks.diffmean_and_probe_layer == 6
        assert cfg.layer_hooks.diffmean_and_probe_layer < 12, (
            "must be a valid block index on the 12-layer decoder"
        )
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert isinstance(dataclass_cfg, AxBenchBaseConfig)
        assert dataclass_cfg.neuronpedia_source == "22-gemmascope-2-res-65k"
        # Passes both model-identity checks in validate_config: the tokenizer
        # (google/gemma-3-4b-pt matches the gemma-3-4b-pt model) and the SAE
        # model in feature_dir (m_gemma-3-4b-pt) matching that extraction model.
        validate_config(dataclass_cfg)

    @pytest.mark.parametrize(
        "tier,expected_max", [("dev", 5), ("standard", 50), ("full", 500)]
    )
    def test_all_three_tiers_yield_valid_config(
        self, tier: str, expected_max: int
    ) -> None:
        """Parameterized over dev/standard/full — every tier composes
        and the resulting dataclass passes validation."""
        cfg = _compose_root(tier=tier)
        assert cfg.max_concepts == expected_max
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert isinstance(dataclass_cfg, AxBenchBaseConfig)
        assert dataclass_cfg.max_concepts == expected_max

    def test_tier_dev_sets_max_concepts_5(self) -> None:
        cfg = _compose_root(tier="dev")
        assert cfg.max_concepts == 5

    def test_tier_standard_sets_max_concepts_50(self) -> None:
        cfg = _compose_root(tier="standard")
        assert cfg.max_concepts == 50

    def test_tier_full_sets_max_concepts_500(self) -> None:
        cfg = _compose_root(tier="full")
        assert cfg.max_concepts == 500

    def test_stages_all_sets_stage_all(self) -> None:
        cfg = _compose_root(stages="all")
        assert cfg.stage == "all"

    def test_methods_all_sets_enabled_methods_none(self) -> None:
        """methods=all means 'run every method in the registry' — the
        registry treats ``enabled_methods=None`` as that."""
        cfg = _compose_root(methods="all")
        assert cfg.enabled_methods is None


# ---------------------------------------------------------------------------
# Mandatory-field enumeration
# ---------------------------------------------------------------------------


class TestMandatoryFields:
    """Every mandatory ``AxBenchBaseConfig`` field must be either
    explicitly set in the root YAML or resolvable via compose+override."""

    def test_all_required_fields_present(self) -> None:
        """After composing without overriding mandatory fields, the
        DictConfig has all six mandatory keys (each either MISSING
        or set)."""
        GlobalHydra.instance().clear()
        with initialize_config_dir(
            config_dir=str(_CONFIG_DIR), version_base=None
        ):
            cfg = compose(
                config_name="axbench_base",
                overrides=["tiers=standard", "stages=all", "methods=all"],
            )
        present_keys = set(cfg.keys())
        for key in (
            "feature_dir",
            "feature_statistics_path",
            "decoder_only",
            "encdec",
            "device",
            "output_root",
        ):
            assert key in present_keys


# ---------------------------------------------------------------------------
# OmegaConf -> dataclass conversion
# ---------------------------------------------------------------------------


class TestConfigConvertsToDataclass:
    """cfg_to_dataclass returns a well-formed AxBenchBaseConfig."""

    def test_config_converts_to_dataclass(self) -> None:
        cfg = _compose_root()
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert isinstance(dataclass_cfg, AxBenchBaseConfig)
        assert dataclass_cfg.feature_dir == "out/data/features/test"
        assert (
            dataclass_cfg.feature_statistics_path
            == "out/data/features/test/stats.safetensors"
        )
        assert dataclass_cfg.decoder_only == {
            "source": "base",
            "model_tag": "d13",
            "step": 10000,
        }
        assert dataclass_cfg.encdec == {
            "source": "encdec",
            "model_tag": "d13",
            "step": 10000,
        }
        assert dataclass_cfg.device == "cuda"
        assert dataclass_cfg.output_root is None
        assert dataclass_cfg.max_concepts == 50
        assert dataclass_cfg.stage == "all"
        assert dataclass_cfg.enabled_methods is None
        assert dataclass_cfg.neuronpedia_source == "8-gemmascope-res-16k"
        assert dataclass_cfg.neuronpedia_model_id == "gemma-2-2b"
        assert dataclass_cfg.feature_extraction_model_id is None
        assert dataclass_cfg.extraction_tokenizer_id == "google/gemma-2-2b"
        assert dataclass_cfg.positive_max_attempts_per_seed == 8
        assert dataclass_cfg.positive_zero_hit_bail_after == 2000
        assert dataclass_cfg.conditioning is not None
        assert dataclass_cfg.conditioning["context_cache_dir"] is None
        assert dataclass_cfg.conditioning["n_sae_features"] == 16384
        assert dataclass_cfg.conditioning["num_buckets"] == 2

    def test_semantic_default_changes_retag_run_oracle(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Semantic QC/retry defaults must not reuse a legacy Stage 1 run tag."""
        fixed = datetime(2026, 5, 5, 12, 34, 56, tzinfo=UTC)

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return fixed

        monkeypatch.setattr(axbench_cli, "_DateTime", _FrozenDateTime)
        dataclass_cfg = cfg_to_dataclass(_compose_root())

        assert dataclass_cfg.output_root is None
        assert dataclass_cfg.accept_threshold == 0.3
        assert axbench_cli._generate_run_tag(dataclass_cfg) == (
            "20260505-123456-b50de4cb"
        )

    def test_positive_refill_cli_overrides_compose(self) -> None:
        cfg = cfg_to_dataclass(
            _compose_root(
                extra=[
                    "positive_max_attempts_per_seed=3",
                    "positive_zero_hit_bail_after=null",
                ]
            )
        )

        assert cfg.positive_max_attempts_per_seed == 3
        assert cfg.positive_zero_hit_bail_after is None

    def test_validation_shards_default_is_369(self) -> None:
        """The root YAML doesn't need to set this, but the resulting
        dataclass should have ``[369]`` (the dataclass default)."""
        cfg = _compose_root()
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.validation_shards == [369]

    def test_nested_configs_instantiated(self) -> None:
        """The nested dataclass fields must be real dataclass instances,
        not dicts or DictConfigs."""
        cfg = _compose_root()
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert isinstance(dataclass_cfg.synth, SynthConfig)
        assert isinstance(dataclass_cfg.judge, JudgeConfig)
        assert isinstance(dataclass_cfg.generation, GenerationConfig)
        assert isinstance(dataclass_cfg.layer_hooks, LayerHooksConfig)
        assert isinstance(dataclass_cfg.lora, LoraConfig)
        assert isinstance(
            dataclass_cfg.concept_examples_qc_judge, LLMClientConfig
        )

    def test_config_rejects_unknown_keys(self) -> None:
        """OmegaConf struct mode rejects adding unknown keys."""
        cfg = _compose_root()
        OmegaConf.set_struct(cfg, True)
        with pytest.raises(ConfigAttributeError):
            cfg.this_key_is_not_real = "boom"

    def test_mandatory_field_unset_raises_on_convert(self) -> None:
        """Compose the root config without the feature_dir override and
        try to convert — should raise AxBenchBaseError."""
        GlobalHydra.instance().clear()
        with initialize_config_dir(
            config_dir=str(_CONFIG_DIR), version_base=None
        ):
            cfg = compose(
                config_name="axbench_base",
                overrides=[
                    "tiers=standard",
                    "stages=all",
                    "methods=all",
                    # intentionally omit feature_dir override
                ],
            )
        with pytest.raises(AxBenchBaseError, match="feature_dir"):
            cfg_to_dataclass(cfg)


# ---------------------------------------------------------------------------
# Explicit tier/stage/method overrides via CLI-style overrides
# ---------------------------------------------------------------------------


class TestOverridesComposePath:
    """Override any field on the command line and it propagates through
    the compose path into the dataclass."""

    def test_concept_feature_mode_override_via_cli(self) -> None:
        cfg = _compose_root(extra=["concept_feature_mode=delta_ranked"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.concept_feature_mode == "delta_ranked"

    def test_run_negative_mode_override_via_cli(self) -> None:
        cfg = _compose_root(extra=["run_negative_mode=false"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.run_negative_mode is False

    def test_validation_shards_override_via_cli(self) -> None:
        cfg = _compose_root(extra=["validation_shards=[1,2,3]"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.validation_shards == [1, 2, 3]

    def test_stage_override_via_cli(self) -> None:
        cfg = _compose_root(extra=["stage=prepare"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.stage == "prepare"

    def test_enabled_methods_override_via_cli(self) -> None:
        cfg = _compose_root(extra=["enabled_methods=[diffmean,probe]"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.enabled_methods == ["diffmean", "probe"]

    def test_max_concepts_override_via_cli(self) -> None:
        cfg = _compose_root(tier="standard", extra=["max_concepts=7"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.max_concepts == 7

    def test_profiling_cap_overrides_via_cli(self) -> None:
        cfg = _compose_root(
            tier="dev",
            extra=[
                "concept_examples_qc_target_surviving=5",
                "concept_description_fetch_limit=50",
                "verification_candidate_scan_limit=1000",
                "verification_candidate_attempt_limit=300",
            ],
        )
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.concept_examples_qc_target_surviving == 5
        assert dataclass_cfg.concept_description_fetch_limit == 50
        assert dataclass_cfg.verification_candidate_scan_limit == 1000
        assert dataclass_cfg.verification_candidate_attempt_limit == 300

    def test_verification_zero_hit_bail_after_default_from_yaml(self) -> None:
        """The root YAML surfaces the field with default 2000, matching
        the dataclass default. Default compose with no override picks
        up that value through cfg_to_dataclass."""
        cfg = _compose_root()
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_zero_hit_bail_after == 2000

    def test_verification_zero_hit_bail_after_override_via_cli(self) -> None:
        """CLI override sets a smaller bail threshold."""
        cfg = _compose_root(extra=["verification_zero_hit_bail_after=500"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_zero_hit_bail_after == 500

    def test_verification_zero_hit_bail_after_null_disables_via_cli(self) -> None:
        """Setting null on the CLI disables the bailout (legacy behavior)."""
        cfg = _compose_root(extra=["verification_zero_hit_bail_after=null"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_zero_hit_bail_after is None

    def test_verification_api_model_defaults_to_ling_flash(self) -> None:
        cfg = _compose_root()
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_api_model == "inclusionai/ling-2.6-flash"

    def test_verification_api_model_override_via_cli(self) -> None:
        cfg = _compose_root(
            extra=["verification_api_model=openai/gpt-5-nano"],
        )
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_api_model == "openai/gpt-5-nano"

    def test_verification_api_model_can_be_reset_to_null_via_cli(self) -> None:
        cfg = _compose_root(extra=["verification_api_model=null"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_api_model is None

    def test_synth_defaults_to_high_throughput_limits(self) -> None:
        cfg = _compose_root()
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.synth.model == "inclusionai/ling-2.6-flash"
        assert dataclass_cfg.synth.requests_per_minute == 6_000
        assert dataclass_cfg.synth.max_concurrent == 100

    def test_synth_throughput_limits_override_via_cli(self) -> None:
        cfg = _compose_root(
            extra=[
                "synth.requests_per_minute=60",
                "synth.max_concurrent=10",
            ],
        )
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.synth.requests_per_minute == 60
        assert dataclass_cfg.synth.max_concurrent == 10

    def test_verification_api_prefix_wave_size_default_from_yaml(self) -> None:
        cfg = _compose_root()
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_api_prefix_wave_size == 100

    def test_verification_api_prefix_wave_size_override_via_cli(self) -> None:
        cfg = _compose_root(extra=["verification_api_prefix_wave_size=32"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_api_prefix_wave_size == 32

    def test_stage0c_multi_gpu_block_size_default_from_yaml(self) -> None:
        """The root YAML surfaces the field with default 16, matching
        the dataclass default. Default compose with no override picks
        up that value through cfg_to_dataclass."""
        cfg = _compose_root()
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.stage0c_multi_gpu_block_size == 16

    def test_stage0c_multi_gpu_block_size_override_via_cli(self) -> None:
        cfg = _compose_root(extra=["stage0c_multi_gpu_block_size=64"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.stage0c_multi_gpu_block_size == 64

    def test_verification_cross_prefix_batch_size_default_from_yaml(self) -> None:
        cfg = _compose_root()
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_cross_prefix_batch_size == 4

    def test_verification_cross_prefix_batch_size_override_via_cli(self) -> None:
        cfg = _compose_root(extra=["verification_cross_prefix_batch_size=8"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_cross_prefix_batch_size == 8

    def test_verification_prefill_mode_default_from_yaml(self) -> None:
        cfg = _compose_root()
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_prefill_mode == "shared"

    def test_verification_prefill_mode_override_via_cli(self) -> None:
        cfg = _compose_root(extra=["verification_prefill_mode=repeated"])
        dataclass_cfg = cfg_to_dataclass(cfg)
        assert dataclass_cfg.verification_prefill_mode == "repeated"

    def test_removed_batched_generation_mode_override_fails_loudly(self) -> None:
        with pytest.raises(
            ConfigCompositionException,
            match="batched_generation_mode",
        ):
            _compose_root(extra=["batched_generation_mode=native"])


# ---------------------------------------------------------------------------
# Direct config_loader smoke tests
# ---------------------------------------------------------------------------


class TestConfigLoaderDirect:
    """Unit tests for ``cfg_to_dataclass`` without going through Hydra."""

    def _minimal_cfg(self) -> DictConfig:
        return OmegaConf.create(
            {
                "feature_dir": "out/data/features/test",
                "feature_statistics_path": "stats.safetensors",
                "decoder_only": {
                    "source": "base",
                    "model_tag": "d13",
                    "step": 10000,
                },
                "encdec": {
                    "source": "encdec",
                    "model_tag": "d13",
                    "step": 10000,
                },
                "device": "cuda",
                "output_root": "out/eval/axbench_base",
            }
        )

    def test_minimal_cfg_yields_defaults(self) -> None:
        dc = cfg_to_dataclass(self._minimal_cfg())
        assert dc.validation_shards == [369]
        assert dc.concept_feature_mode == "canonical"
        assert dc.max_concepts == 500
        assert dc.max_retries == 100
        assert dc.synth.transport_max_retries == 3

    def test_nested_overrides_propagate(self) -> None:
        cfg = self._minimal_cfg()
        cfg.judge = OmegaConf.create(
            {"requests_per_minute": 1_000, "parse_max_retries": 7}
        )
        dc = cfg_to_dataclass(cfg)
        assert isinstance(dc.judge, JudgeConfig)
        assert dc.judge.requests_per_minute == 1_000
        assert dc.judge.parse_max_retries == 7
        # Non-overridden fields retain their defaults.
        assert dc.judge.provider == "openrouter"

    def test_synth_transport_retry_override_propagates_independently(self) -> None:
        cfg = self._minimal_cfg()
        cfg.max_retries = 41
        cfg.synth = OmegaConf.create(
            {"transport_max_retries": 2, "max_concurrent": 17}
        )
        dc = cfg_to_dataclass(cfg)
        assert isinstance(dc.synth, SynthConfig)
        assert dc.max_retries == 41
        assert dc.synth.transport_max_retries == 2
        assert dc.synth.max_concurrent == 17
        assert dc.synth.model == "inclusionai/ling-2.6-flash"

    def test_layer_hooks_override(self) -> None:
        cfg = self._minimal_cfg()
        cfg.layer_hooks = OmegaConf.create({"diffmean_and_probe_layer": 12})
        dc = cfg_to_dataclass(cfg)
        assert dc.layer_hooks.diffmean_and_probe_layer == 12

    def test_feature_extraction_model_id_override(self) -> None:
        cfg = _compose_root(
            extra=[
                "neuronpedia_model_id=gemma-3-4b",
                "feature_extraction_model_id=gemma-3-4b-pt",
                "neuronpedia_source=22-gemmascope-2-res-65k",
                "extraction_tokenizer_id=google/gemma-3-4b-pt",
                "layer_hooks.diffmean_and_probe_layer=22",
            ],
        )
        dc = cfg_to_dataclass(cfg)
        assert dc.neuronpedia_model_id == "gemma-3-4b"
        assert dc.feature_extraction_model_id == "gemma-3-4b-pt"
        assert dc.neuronpedia_source == "22-gemmascope-2-res-65k"
        assert dc.extraction_tokenizer_id == "google/gemma-3-4b-pt"
        assert dc.layer_hooks.diffmean_and_probe_layer == 22

    def test_lora_override(self) -> None:
        cfg = self._minimal_cfg()
        cfg.lora = OmegaConf.create({"rank": 16, "lr": 5e-5})
        dc = cfg_to_dataclass(cfg)
        assert dc.lora.rank == 16
        assert dc.lora.lr == 5e-5
        assert dc.lora.alpha == 16.0  # default retained

    def test_generation_override(self) -> None:
        cfg = self._minimal_cfg()
        cfg.generation = OmegaConf.create(
            {"max_new_tokens": 128, "temperature": 0.9}
        )
        dc = cfg_to_dataclass(cfg)
        assert dc.generation.max_new_tokens == 128
        assert dc.generation.temperature == 0.9

    def test_concept_examples_qc_judge_override(self) -> None:
        cfg = self._minimal_cfg()
        cfg.concept_examples_qc_judge = OmegaConf.create(
            {"model": "openai/gpt-4o"}
        )
        dc = cfg_to_dataclass(cfg)
        assert dc.concept_examples_qc_judge.model == "openai/gpt-4o"
        assert dc.concept_examples_qc_judge.provider == "openrouter"

    def test_profile_override(self) -> None:
        cfg = self._minimal_cfg()
        cfg.profile = OmegaConf.create(
            {
                "enabled": True,
                "output_dir": "custom-profile",
                "gpu_poll_interval_ms": 25,
            }
        )
        dc = cfg_to_dataclass(cfg)
        assert isinstance(dc.profile, ProfileConfig)
        assert dc.profile.enabled is True
        assert dc.profile.output_dir == "custom-profile"
        assert dc.profile.gpu_poll_interval_ms == 25
        assert dc.profile.gpu_utilization is True

    def test_missing_mandatory_raises(self) -> None:
        cfg = self._minimal_cfg()
        cfg.feature_dir = "???"
        with pytest.raises(AxBenchBaseError, match="feature_dir"):
            cfg_to_dataclass(cfg)

    def test_missing_nested_field_raises(self) -> None:
        cfg = self._minimal_cfg()
        cfg.decoder_only.source = "???"
        with pytest.raises(AxBenchBaseError, match="decoder_only"):
            cfg_to_dataclass(cfg)

    def test_unknown_top_level_key_raises(self) -> None:
        """Passing an unknown key to ``AxBenchBaseConfig`` surfaces as an
        ``AxBenchBaseError`` — the dataclass constructor would raise
        ``TypeError`` and ``cfg_to_dataclass`` wraps that."""
        cfg = self._minimal_cfg()
        cfg.this_key_is_not_real = "boom"
        with pytest.raises(AxBenchBaseError, match="this_key_is_not_real"):
            cfg_to_dataclass(cfg)

    def test_accepts_plain_dict(self) -> None:
        """The converter also accepts plain dicts (not just
        DictConfig) so callers can build the dict however they like."""
        data = OmegaConf.to_container(
            self._minimal_cfg(), resolve=True, throw_on_missing=True
        )
        assert isinstance(data, dict)
        # Round-trip through OmegaConf.create so the converter sees a
        # DictConfig (the public API takes a DictConfig).
        dc = cfg_to_dataclass(OmegaConf.create(data))
        assert dc.feature_dir == "out/data/features/test"

    def test_none_nested_config_falls_back_to_default(self) -> None:
        """Setting a nested config to ``None`` must drop the key so
        the dataclass default factory builds the default."""
        cfg = self._minimal_cfg()
        cfg.judge = None
        dc = cfg_to_dataclass(cfg)
        assert isinstance(dc.judge, JudgeConfig)
        assert dc.judge.provider == "openrouter"

    def test_non_dict_nested_raises(self) -> None:
        """Passing a scalar where a dict is expected must raise
        ``AxBenchBaseError``, not a confusing ``TypeError``."""
        cfg = self._minimal_cfg()
        # OmegaConf lets us set the nested field to a string.
        cfg.judge = "not-a-dict"
        with pytest.raises(AxBenchBaseError, match="judge"):
            cfg_to_dataclass(cfg)

    def test_nested_config_with_unknown_subkey_raises(self) -> None:
        """Unknown subkey on a nested config surfaces as an
        ``AxBenchBaseError`` pointing at the offending nested field."""
        cfg = self._minimal_cfg()
        cfg.lora = OmegaConf.create(
            {"bogus_lora_field": 42}
        )
        with pytest.raises(AxBenchBaseError, match="lora"):
            cfg_to_dataclass(cfg)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_concepts", 0),
            ("seed_pool_train_ratio", 1.0),
            ("validation_shards", [1, 1]),
        ],
    )
    def test_refined_top_level_domain_errors_are_wrapped_at_ingress(
        self,
        field: str,
        value: object,
    ) -> None:
        cfg = self._minimal_cfg()
        cfg[field] = value
        with pytest.raises(AxBenchBaseError, match=field) as exc_info:
            cfg_to_dataclass(cfg)
        assert isinstance(
            exc_info.value.__cause__,
            BeartypeCallHintParamViolation,
        )

    def test_refined_nested_domain_errors_are_wrapped_at_ingress(self) -> None:
        cfg = self._minimal_cfg()
        cfg.synth = OmegaConf.create({"transport_max_retries": -1})
        with pytest.raises(AxBenchBaseError, match="synth") as exc_info:
            cfg_to_dataclass(cfg)
        assert isinstance(
            exc_info.value.__cause__,
            BeartypeCallHintParamViolation,
        )

    def test_rejects_list_config(self) -> None:
        """A top-level ListConfig is not a valid AxBenchBaseConfig
        input — the converter must reject it with
        ``AxBenchBaseError``."""
        list_cfg = OmegaConf.create([1, 2, 3])
        with pytest.raises(AxBenchBaseError, match="dict-like"):
            cfg_to_dataclass(cast(DictConfig, list_cfg))
