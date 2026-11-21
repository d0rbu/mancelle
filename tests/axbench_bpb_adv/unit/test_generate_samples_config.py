"""Config-wiring tests for the generate-samples DictConfig adapter.

Pins the seam ``run_generate_samples_from_dictconfig`` -> ``run_generate_samples``:
the ``generation:`` block parses into a ``GenerationConfig``, the top-level
``counter_direction_family`` is peeled off and threaded through, ``run_tag``
defaults to ``"samples"`` when unset, and the committed YAML composes through
Hydra into a config the parser accepts (model-loading is stubbed). Also pins the
``generation:`` defaults so the committed knobs cannot silently drift.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

import steering.eval.axbench_bpb_adv.sample_generation as sg
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_bpb.types import LoraParams
from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.sample_generation import (
    GenerationConfig,
    run_generate_samples,
    run_generate_samples_from_dictconfig,
)

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "eval" / "axbench_bpb_adv"


def _base_overrides(tmp_path: Path) -> dict[str, Any]:
    return {
        "source_run_dir": str(tmp_path / "src"),
        "output_root": str(tmp_path / "out"),
        "device": "cpu",
        "feature_dir": "s256/l6/a_res/p_mean/f16384/m_gemma-2-2b/sft",
        "feature_statistics_path": str(tmp_path / "stats.safetensors"),
    }


def test_dictconfig_threads_generation_and_counter_family(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _spy_run(
        cfg: AxBenchBpbAdvConfig,
        *,
        generation: GenerationConfig,
        counter_direction_family: str | None = None,
    ) -> list[Path]:
        captured["cfg"] = cfg
        captured["generation"] = generation
        captured["counter_direction_family"] = counter_direction_family
        return []

    monkeypatch.setattr(sg, "run_generate_samples", _spy_run)

    cfg = OmegaConf.create(
        {
            **_base_overrides(tmp_path),
            "target_models": ["decoder_only"],
            "counter_direction_family": "linear_probe",
            "generation": {
                "n_concepts": 2,
                "n_prefixes_per_concept": 3,
                "n_completions": 4,
                "temperature": 0.5,
                "top_k": None,
                "max_new_tokens": 16,
                "base_seed": 7,
                "methods": ["diffmean_decoder_only"],
            },
        }
    )

    run_generate_samples_from_dictconfig(cfg)

    generation = captured["generation"]
    assert isinstance(generation, GenerationConfig)
    assert generation.n_concepts == 2
    assert generation.n_prefixes_per_concept == 3
    assert generation.n_completions == 4
    assert generation.temperature == 0.5
    assert generation.top_k is None
    assert generation.max_new_tokens == 16
    assert generation.base_seed == 7
    assert generation.methods == ["diffmean_decoder_only"]
    # Top-level counter_direction_family is peeled off and threaded through.
    assert captured["counter_direction_family"] == "linear_probe"
    # run_tag defaults to "samples" when unset.
    assert captured["cfg"].run_tag == "samples"


def test_dictconfig_default_generation_block(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _spy_run(
        cfg: AxBenchBpbAdvConfig,
        *,
        generation: GenerationConfig,
        counter_direction_family: str | None = None,
    ) -> list[Path]:
        captured["generation"] = generation
        captured["counter_direction_family"] = counter_direction_family
        return []

    monkeypatch.setattr(sg, "run_generate_samples", _spy_run)

    cfg = OmegaConf.create({**_base_overrides(tmp_path), "target_models": ["decoder_only"]})
    run_generate_samples_from_dictconfig(cfg)

    # Absent generation block -> the GenerationConfig defaults.
    assert captured["generation"] == GenerationConfig()
    # Absent top-level counter_direction_family -> None (scorer-resolved downstream).
    assert captured["counter_direction_family"] is None


def test_committed_yaml_composes_and_parses(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _spy_run(
        cfg: AxBenchBpbAdvConfig,
        *,
        generation: GenerationConfig,
        counter_direction_family: str | None = None,
    ) -> list[Path]:
        captured["cfg"] = cfg
        captured["generation"] = generation
        captured["counter_direction_family"] = counter_direction_family
        return []

    monkeypatch.setattr(sg, "run_generate_samples", _spy_run)

    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="generate_samples",
            overrides=[
                f"source_run_dir={tmp_path / 'src'}",
                f"output_root={tmp_path / 'out'}",
                "device=cpu",
                "feature_dir=s256/l6/a_res/p_mean/f16384/m_gemma-2-2b/sft",
                f"feature_statistics_path={tmp_path / 'stats.safetensors'}",
            ],
        )
        assert "roundtrip_steering" not in cfg
        run_generate_samples_from_dictconfig(cfg)

    # The committed generation: defaults survive the round-trip.
    generation = captured["generation"]
    assert generation.n_concepts == 5
    assert generation.n_prefixes_per_concept == 1
    assert generation.n_completions == 5
    assert generation.temperature == pytest.approx(0.7)
    assert generation.top_k == 50
    assert generation.max_new_tokens == 128
    assert generation.base_seed == 0
    assert generation.methods is None
    # The committed YAML no longer hardcodes a top-level counter_direction_family;
    # it is opt-in (null), so the scorer-resolved (additive-config) value is used
    # downstream. None here means "resolve from the eval's additive config".
    assert captured["counter_direction_family"] is None
    # Decoder-only-only target by default.
    assert captured["cfg"].target_models == ["decoder_only"]
    # Generate-samples defaults mirror the eval's default metric.
    assert captured["cfg"].steering_metric == "roundtrip"
    # The shared AxBench-BPB-Adv config loader supplies this internal default;
    # generate_samples.yaml does not need to restate it.
    assert captured["cfg"].roundtrip_steering.enabled is True


def test_committed_yaml_accepts_legacy_metric_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _spy_run(
        cfg: AxBenchBpbAdvConfig,
        *,
        generation: GenerationConfig,
        counter_direction_family: str | None = None,
    ) -> list[Path]:
        captured["cfg"] = cfg
        captured["generation"] = generation
        captured["counter_direction_family"] = counter_direction_family
        return []

    monkeypatch.setattr(sg, "run_generate_samples", _spy_run)

    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="generate_samples",
            overrides=[
                f"source_run_dir={tmp_path / 'src'}",
                f"output_root={tmp_path / 'out'}",
                "device=cpu",
                "feature_dir=s256/l6/a_res/p_mean/f16384/m_gemma-2-2b/sft",
                f"feature_statistics_path={tmp_path / 'stats.safetensors'}",
                "steering_metric=bpb_class_margin",
            ],
        )
        assert "roundtrip_steering" not in cfg
        run_generate_samples_from_dictconfig(cfg)

    assert captured["cfg"].steering_metric == "bpb_class_margin"
    # Legacy sample generation only needs the metric selector. It does not need
    # to mirror score-stage roundtrip_steering.enabled=false.
    assert captured["cfg"].roundtrip_steering.enabled is True


def test_committed_yaml_force_added_disabled_roundtrip_block_keeps_guard(
    tmp_path: Path,
) -> None:
    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="generate_samples",
            overrides=[
                f"source_run_dir={tmp_path / 'src'}",
                f"output_root={tmp_path / 'out'}",
                "device=cpu",
                "feature_dir=s256/l6/a_res/p_mean/f16384/m_gemma-2-2b/sft",
                f"feature_statistics_path={tmp_path / 'stats.safetensors'}",
                "+roundtrip_steering.enabled=false",
            ],
        )

    with pytest.raises(AxBenchBpbAdvError, match="roundtrip_steering.enabled=false"):
        run_generate_samples_from_dictconfig(cfg)


def test_committed_yaml_legacy_metric_accepts_force_added_disabled_roundtrip_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _spy_run(
        cfg: AxBenchBpbAdvConfig,
        *,
        generation: GenerationConfig,
        counter_direction_family: str | None = None,
    ) -> list[Path]:
        captured["cfg"] = cfg
        return []

    monkeypatch.setattr(sg, "run_generate_samples", _spy_run)

    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="generate_samples",
            overrides=[
                f"source_run_dir={tmp_path / 'src'}",
                f"output_root={tmp_path / 'out'}",
                "device=cpu",
                "feature_dir=s256/l6/a_res/p_mean/f16384/m_gemma-2-2b/sft",
                f"feature_statistics_path={tmp_path / 'stats.safetensors'}",
                "steering_metric=bpb_class_margin",
                "+roundtrip_steering.enabled=false",
            ],
        )
        run_generate_samples_from_dictconfig(cfg)

    assert captured["cfg"].steering_metric == "bpb_class_margin"
    assert captured["cfg"].roundtrip_steering.enabled is False


# ---------------------------------------------------------------------------
# F3: an absent/None top-level counter_direction_family resolves from the eval's
# additive config exactly as the scorer does -- it is NOT silently forced to
# "diffmean". An EXPLICIT top-level value still overrides.
# ---------------------------------------------------------------------------


def _stub_model_loading(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the model-loading boundary so the resolution path runs model-free.

    Captures the kwargs threaded into ``run_generate_samples_core`` (the seam the
    resolved ``counter_direction_family`` flows through) without loading any model.
    """

    captured: dict[str, Any] = {}

    class _StubRuntimeState:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def decoder_only(self) -> Any:
            return SimpleNamespace(model=object(), tokenizer=object())

        def target_device(self) -> str:
            return "cpu"

        def offload_models(self) -> None:
            pass

    def _stub_core(**kwargs: Any) -> list[Path]:
        captured["counter_direction_family"] = kwargs["counter_direction_family"]
        return []

    monkeypatch.setattr(sg, "RuntimeState", _StubRuntimeState)
    monkeypatch.setattr(sg, "detect_target_model_metadata", lambda *a, **k: None)
    monkeypatch.setattr(sg, "run_generate_samples_core", _stub_core)
    return captured


def _install_source_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_root: Path,
    run_tag: str = "run",
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag)
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    payload = sg.BpbAdvSourceContract(
        seed_fingerprint="f" * 64,
        source_run_dir="source",
    ).to_json_dict()
    paths.source_contract().write_text(json.dumps(payload), encoding="utf-8")
    pq.write_table(
        pa.Table.from_pylist([], schema=sg.CANDIDATES_SCHEMA),
        paths.candidates_path(),
    )
    validation = SimpleNamespace(
        contract=SimpleNamespace(to_json_dict=lambda: payload),
        require_directions=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sg,
        "validate_stage_0_3_source_run",
        lambda *args, **kwargs: validation,
    )


def test_sample_direction_mismatch_fails_before_runtime_state_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _faster_gcg_cfg(tmp_path)
    _install_source_preflight(monkeypatch, output_root=tmp_path / "out")
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    payload = json.loads(paths.source_contract().read_text(encoding="utf-8"))

    def reject_directions(*_args: Any, **_kwargs: Any) -> None:
        raise AxBenchBpbAdvError(
            "direction artifact layer mismatch: requested/applied layer=6, "
            "artifact training layer=10"
        )

    monkeypatch.setattr(
        sg,
        "validate_stage_0_3_source_run",
        lambda *args, **kwargs: SimpleNamespace(
            contract=SimpleNamespace(to_json_dict=lambda: payload),
            require_directions=reject_directions,
        ),
    )
    monkeypatch.setattr(
        sg,
        "RuntimeState",
        lambda *_args, **_kwargs: pytest.fail(
            "sample generation must reject before target-model runtime construction"
        ),
    )

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        run_generate_samples(cfg, generation=GenerationConfig())

    message = str(exc_info.value)
    assert "requested/applied layer=6" in message
    assert "artifact training layer=10" in message


def _additive_cfg(*, counter_direction_family: str, tmp_path: Path) -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "src"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        feature_dir="s256/l6/a_res/p_mean/f16384/m_gemma-2-2b/sft",
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        device="cpu",
        adversarial_methods=[
            AdditiveDirectionConfig(counter_direction_family=cast(Any, counter_direction_family))
        ],
    )


def test_counter_family_none_resolves_from_additive_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The additive config says linear_probe; NO top-level override is given (None).
    # Generation must resolve linear_probe (the scorer's value), NOT diffmean.
    captured = _stub_model_loading(monkeypatch)
    cfg = _additive_cfg(counter_direction_family="linear_probe", tmp_path=tmp_path)
    _install_source_preflight(monkeypatch, output_root=tmp_path / "out")

    run_generate_samples(cfg, generation=GenerationConfig(), counter_direction_family=None)

    assert captured["counter_direction_family"] == "linear_probe"


def test_explicit_top_level_counter_family_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An EXPLICIT top-level value still wins over the additive config's value.
    captured = _stub_model_loading(monkeypatch)
    cfg = _additive_cfg(counter_direction_family="linear_probe", tmp_path=tmp_path)
    _install_source_preflight(monkeypatch, output_root=tmp_path / "out")

    run_generate_samples(cfg, generation=GenerationConfig(), counter_direction_family="diffmean")

    assert captured["counter_direction_family"] == "diffmean"


# ---------------------------------------------------------------------------
# F4: a scalar `methods` (a bare string instead of a list) must fail LOUD --
# before the fix `[str(m) for m in methods]` iterated the string's CHARACTERS,
# silently building a per-character method whitelist that matched nothing (zero
# output, no error).
# ---------------------------------------------------------------------------


def test_generation_methods_scalar_string_raises() -> None:
    # A YAML scalar `methods: diffmean_decoder_only` arrives as a bare str.
    with pytest.raises(AxBenchBpbAdvError, match="methods"):
        sg._generation_config_from_mapping({"methods": "diffmean_decoder_only"})


def test_generation_methods_scalar_via_dictconfig_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same scalar through the full Hydra entry hits the friendly error path
    # (AxBenchBpbAdvError), never a silent zero-output run.
    _stub_model_loading(monkeypatch)
    cfg = OmegaConf.create(
        {
            **_base_overrides(tmp_path),
            "target_models": ["decoder_only"],
            "generation": {"methods": "diffmean_decoder_only"},
        }
    )
    with pytest.raises(AxBenchBpbAdvError, match="methods"):
        run_generate_samples_from_dictconfig(cfg)


def test_generation_methods_list_is_accepted() -> None:
    # The valid case: a list survives and becomes the whitelist (no regression).
    generation = sg._generation_config_from_mapping(
        {"methods": ["diffmean_decoder_only", "control_decoder_only"]}
    )
    assert generation.methods == ["diffmean_decoder_only", "control_decoder_only"]


# ---------------------------------------------------------------------------
# F8: an ABSENT top_k must fall back to the GenerationConfig field default (50),
# NOT None (unbounded). An explicit top_k <= 0 is treated as disabled (None) so a
# stray top_k: 0 never reaches torch.topk(0).
# ---------------------------------------------------------------------------


def test_generation_absent_top_k_uses_field_default_not_none() -> None:
    # No top_k key -> the GenerationConfig field default (50), not None.
    generation = sg._generation_config_from_mapping({"n_concepts": 2})
    assert generation.top_k == GenerationConfig().top_k == 50


def test_generation_explicit_null_top_k_disables() -> None:
    # An explicit top_k: null is honoured as disabled (None).
    generation = sg._generation_config_from_mapping({"top_k": None})
    assert generation.top_k is None


def test_generation_top_k_zero_becomes_none() -> None:
    # top_k: 0 must become None (disabled), never reach torch.topk(0).
    generation = sg._generation_config_from_mapping({"top_k": 0})
    assert generation.top_k is None


def test_generation_negative_top_k_becomes_none() -> None:
    # A negative top_k is likewise treated as disabled.
    generation = sg._generation_config_from_mapping({"top_k": -3})
    assert generation.top_k is None


def test_generation_positive_top_k_is_kept() -> None:
    # A positive top_k passes through unchanged (no regression).
    generation = sg._generation_config_from_mapping({"top_k": 20})
    assert generation.top_k == 20


# ---------------------------------------------------------------------------
# Explicit counter-family validation: the top-level counter_direction_family
# override is peeled off BEFORE cfg_to_dataclass, bypassing validate_config's
# {"diffmean","linear_probe"} restriction. A bogus explicit value must raise.
# ---------------------------------------------------------------------------


def test_explicit_bogus_counter_family_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_model_loading(monkeypatch)
    cfg = OmegaConf.create(
        {
            **_base_overrides(tmp_path),
            "target_models": ["decoder_only"],
            "counter_direction_family": "bogus_family",
        }
    )
    with pytest.raises(AxBenchBpbAdvError, match="counter_direction_family"):
        run_generate_samples_from_dictconfig(cfg)


def test_explicit_valid_counter_family_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The valid case still threads through (no regression): linear_probe is allowed.
    captured = _stub_model_loading(monkeypatch)
    cfg = OmegaConf.create(
        {
            **_base_overrides(tmp_path),
            "target_models": ["decoder_only"],
            "counter_direction_family": "linear_probe",
        }
    )
    _install_source_preflight(
        monkeypatch,
        output_root=tmp_path / "out",
        run_tag="samples",
    )
    run_generate_samples_from_dictconfig(cfg)
    assert captured["counter_direction_family"] == "linear_probe"


# ---------------------------------------------------------------------------
# F13: cfg.target_models is honoured only for decoder_only (the sole supported
# target today). A non-decoder-only entry (e.g. encdec) must WARN that it is
# skipped -- decoder-only-first, NOT a silent no-op -- and must NOT raise (the
# decoder-only families still generate).
# ---------------------------------------------------------------------------


def test_target_models_encdec_warns_and_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    captured = _stub_model_loading(monkeypatch)
    cfg = _additive_cfg(counter_direction_family="diffmean", tmp_path=tmp_path)
    _install_source_preflight(monkeypatch, output_root=tmp_path / "out")
    # A config that lists BOTH decoder_only and the not-yet-supported encdec.
    cfg.target_models = ["decoder_only", "encdec"]

    with caplog.at_level(logging.WARNING):
        # Must NOT raise: decoder-only still runs (the stubbed core returns []).
        run_generate_samples(cfg, generation=GenerationConfig(), counter_direction_family=None)

    # The skipped non-decoder-only target is named loudly.
    text = caplog.text
    assert "encdec" in text
    # decoder-only still generated (the core was invoked with decoder_only).
    assert captured["counter_direction_family"] == "diffmean"


# ---------------------------------------------------------------------------
# GAP A.3: _resolve_counter_direction_family falls back to "diffmean" when the
# eval config has NO additive method (get_additive_direction_config -> None) and
# the top-level override is None. (The override + additive-present cases are
# already covered above; this pins the bare fallback arm.)
# ---------------------------------------------------------------------------


def _capture_all_core_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub model loading and capture ALL kwargs threaded into the core.

    Like ``_stub_model_loading`` but records the whole kwargs dict (so the LoRA
    bundle + counter_direction_family + offload behaviour can each be asserted),
    and an ``offload_calls`` counter on the RuntimeState ``finally`` path.
    """

    captured: dict[str, Any] = {"offload_calls": 0}

    class _StubRuntimeState:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def decoder_only(self) -> Any:
            return SimpleNamespace(model=object(), tokenizer=object())

        def target_device(self) -> str:
            return "cpu"

        def offload_models(self) -> None:
            captured["offload_calls"] += 1

    def _stub_core(**kwargs: Any) -> list[Path]:
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(sg, "RuntimeState", _StubRuntimeState)
    monkeypatch.setattr(sg, "detect_target_model_metadata", lambda *a, **k: None)
    monkeypatch.setattr(sg, "run_generate_samples_core", _stub_core)
    return captured


def _faster_gcg_cfg(tmp_path: Path) -> AxBenchBpbAdvConfig:
    """A config whose ONLY adversarial method is faster_gcg (no additive)."""

    return AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "src"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        feature_dir="s256/l6/a_res/p_mean/f16384/m_gemma-2-2b/sft",
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        device="cpu",
        adversarial_methods=[FasterGcgConfig()],
        target_models=["decoder_only"],
    )


def test_counter_family_falls_back_to_diffmean_without_additive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_all_core_kwargs(monkeypatch)
    cfg = _faster_gcg_cfg(tmp_path)
    _install_source_preflight(monkeypatch, output_root=tmp_path / "out")

    run_generate_samples(cfg, generation=GenerationConfig(), counter_direction_family=None)

    # No additive method + no override -> the scorer's "diffmean" fallback.
    assert captured["kwargs"]["counter_direction_family"] == "diffmean"


def test_run_generate_samples_forwards_steering_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_all_core_kwargs(monkeypatch)
    cfg = _faster_gcg_cfg(tmp_path)
    cfg.steering_metric = "bpb_class_margin"
    _install_source_preflight(monkeypatch, output_root=tmp_path / "out")

    run_generate_samples(cfg, generation=GenerationConfig())

    assert captured["kwargs"]["steering_metric"] == "bpb_class_margin"


# ---------------------------------------------------------------------------
# GAP A.5: the LoRA bundle is built from cfg.lora_adapter exactly as the scorer
# builds it (LoraParams(cfg=LoraConfig(**cfg.lora_adapter), layer=...)); an empty
# cfg.lora_adapter yields None.
# ---------------------------------------------------------------------------


def test_lora_bundle_built_from_cfg_lora_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_all_core_kwargs(monkeypatch)
    cfg = _faster_gcg_cfg(tmp_path)
    cfg.lora_adapter = {"rank": 4, "alpha": 8.0}
    cfg.diffmean_and_probe_layer = 3
    _install_source_preflight(monkeypatch, output_root=tmp_path / "out")

    run_generate_samples(cfg, generation=GenerationConfig())

    lora = captured["kwargs"]["lora"]
    assert isinstance(lora, LoraParams)
    assert isinstance(lora.cfg, LoraConfig)
    assert lora.cfg.rank == 4
    assert lora.cfg.alpha == 8.0
    # The bundle's layer is the eval's diffmean_and_probe_layer (the scorer's wiring).
    assert lora.layer == 3


def test_empty_lora_adapter_yields_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_all_core_kwargs(monkeypatch)
    cfg = _faster_gcg_cfg(tmp_path)
    # Default cfg.lora_adapter is an empty dict -> no LoRA.
    assert cfg.lora_adapter == {}
    _install_source_preflight(monkeypatch, output_root=tmp_path / "out")

    run_generate_samples(cfg, generation=GenerationConfig())

    assert captured["kwargs"]["lora"] is None


# ---------------------------------------------------------------------------
# GAP A.8 (cheap defensive): run_tag-None raise, offload True vs False in the
# finally, _generation_config_from_mapping non-Mapping + methods-non-Sequence
# raises, and the coercion-failure TypeError->AxBenchBpbAdvError wrapper.
# ---------------------------------------------------------------------------


def test_run_generate_samples_run_tag_none_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # cfg.run_tag is required before sampling (a None run_tag would write under an
    # unstable/ambiguous run dir) -> raise loudly. Reaches the guard before any
    # model loading.
    _capture_all_core_kwargs(monkeypatch)
    cfg = _faster_gcg_cfg(tmp_path)
    cfg.run_tag = None

    with pytest.raises(AxBenchBpbAdvError, match="run_tag must be set"):
        run_generate_samples(cfg, generation=GenerationConfig())


def test_offload_between_stages_true_calls_offload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The finally clause offloads models iff cfg.offload_between_stages is True.
    captured = _capture_all_core_kwargs(monkeypatch)
    cfg = _faster_gcg_cfg(tmp_path)
    cfg.offload_between_stages = True
    _install_source_preflight(monkeypatch, output_root=tmp_path / "out")

    run_generate_samples(cfg, generation=GenerationConfig())

    assert captured["offload_calls"] == 1


def test_offload_between_stages_false_does_not_offload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The negative: offload_between_stages False -> offload_models is NOT called.
    # Teeth: inverting the finally guard would make this RED (offload_calls == 1).
    captured = _capture_all_core_kwargs(monkeypatch)
    cfg = _faster_gcg_cfg(tmp_path)
    cfg.offload_between_stages = False
    _install_source_preflight(monkeypatch, output_root=tmp_path / "out")

    run_generate_samples(cfg, generation=GenerationConfig())

    assert captured["offload_calls"] == 0


def test_generation_config_from_mapping_non_mapping_raises() -> None:
    # A non-None, non-Mapping generation block (e.g. a list) must raise loudly --
    # not be silently treated as defaults or crash on .items().
    with pytest.raises(AxBenchBpbAdvError, match="generation must be a config object"):
        sg._generation_config_from_mapping([1, 2, 3])


def test_generation_methods_int_non_sequence_raises() -> None:
    # GAP A.8: a non-Sequence `methods` value (e.g. an int) hits the same loud
    # raise as the scalar-string case -- it must NOT be silently accepted.
    with pytest.raises(AxBenchBpbAdvError, match="methods must be null or a list"):
        sg._generation_config_from_mapping({"methods": 5})


def test_generation_config_coercion_failure_wrapped() -> None:
    # GAP A.8: a non-coercible scalar (n_concepts="abc") raises ValueError inside
    # the GenerationConfig coercion; it must be WRAPPED into a friendly
    # AxBenchBpbAdvError ("invalid generation config: ..."), not leak the raw
    # ValueError.
    with pytest.raises(AxBenchBpbAdvError, match="invalid generation config"):
        sg._generation_config_from_mapping({"n_concepts": "abc"})
