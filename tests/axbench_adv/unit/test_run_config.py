"""Strict Hydra boundary tests for the AxBench-Adv runner."""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hydra
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.eval.axbench_adv import run as run_module
from scripts.eval.axbench_adv.pressures import Pressure
from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.pipeline import AxBenchAdvRunConfig, _worker_payload


def _raw() -> dict[str, object]:
    return {
        "source": {
            "prepared_dir": "/source/prepared",
            "main_parquet": "/source/main/inference/steering_data.parquet",
            "aux_parquet": "/source/aux/inference/steering_data.parquet",
            "evaluated_jsonl": "/source/evaluated/steering.jsonl",
            "project_parquet": "/source/project/inference/steering_data.parquet",
            "target_meta": "/source/checkpoint/meta_054600.json",
            "feature_statistics": "/source/features/feature_statistics.safetensors",
            "hashes": {
                "main_parquet": "1" * 64,
                "aux_parquet": "2" * 64,
                "evaluated_jsonl": "3" * 64,
                "project_parquet": "8" * 64,
                "project_manifest": "9" * 64,
                "target_receipt": "a" * 64,
            },
            "expectations": {
                "max_concepts": 2,
                "repository_git_commit": "4" * 40,
                "dataset_revision": "5" * 40,
                "target_model": "fake/Qwen",
                "model_revision": None,
                "seed": 42,
                "temperature": 1.0,
                "output_length": 128,
                "package_lock_sha256": "6" * 64,
                "vendored_axbench_upstream_commit": "7" * 40,
                "project_repository_git_commit": "b" * 40,
                "project_package_lock_sha256": "c" * 64,
                "project_target_step": 54600,
                "project_target_model_tag": "l22_main_h100_8rank_db16_v1",
                "config_name_prefix": "qwen2_5_3b",
                "layer": 28,
                "require_clean_worktree": True,
            },
        },
        "output_dir": "/output/run",
        "judge_cache": "/cache/judge.sqlite",
        "reflection_cache": "/cache/reflection.sqlite",
        "cache_only": False,
        "master_seed": 42,
        "gepa_seed": 43,
        "device": "cuda:0",
        "api_base_url": "https://example.test/v1",
        "worker_timeout_seconds": 120.0,
        "worker_max_prompt_tokens": 1024,
        "seed_suffix": "seed suffix",
        "factor_overrides": {"DiffMean": 0.8},
        "reflection": {
            "provider": "openrouter",
            "model": "fake/reflection",
            "requests_per_minute": 60,
            "max_concurrent": 1,
            "transport_retry_budget_seconds": 300.0,
        },
        "reflection_max_retries": 3,
        "reflection_minibatch_size": 3,
    }


def test_build_config_constructs_exact_typed_boundary() -> None:
    config = build_config(OmegaConf.create(_raw()))
    assert config.source_expectations.max_concepts == 2
    assert config.source_expectations.require_clean_worktree is True
    assert config.source_paths.main_parquet == Path("/source/main/inference/steering_data.parquet")
    assert config.output_dir == Path("/output/run")
    assert config.gepa_seed == 43
    assert config.worker_max_prompt_tokens == 1024
    assert config.project_judge_inflight == 1
    assert config.project_judge_workers == 1
    assert config.project_judge_pair_concurrency == 1
    assert config.project_reuse_single_token_mask is False
    assert config.project_bf16_decoder_linears is False
    assert config.project_native_extraction is False
    assert config.reuse_final_judge_worker is False
    assert config.factor_overrides == {"DiffMean": 0.8}
    assert config.reflection.model == "fake/reflection"


def test_project_judge_inflight_is_bounded_and_explicit() -> None:
    raw = _raw()
    raw["project_judge_inflight"] = 2
    assert build_config(OmegaConf.create(raw)).project_judge_inflight == 2
    for invalid in (True, 0, 4, 1.5):
        raw["project_judge_inflight"] = invalid
        with pytest.raises(AxBenchAdvError, match="project_judge_inflight"):
            build_config(OmegaConf.create(raw))


def test_project_judge_pair_concurrency_is_bounded_and_explicit() -> None:
    raw = _raw()
    raw["project_judge_pair_concurrency"] = 2
    assert build_config(OmegaConf.create(raw)).project_judge_pair_concurrency == 2
    for invalid in (True, 0, 3, 1.5):
        raw["project_judge_pair_concurrency"] = invalid
        with pytest.raises(AxBenchAdvError, match="project_judge_pair_concurrency"):
            build_config(OmegaConf.create(raw))


def test_project_judge_workers_requires_bounded_overlap() -> None:
    raw = _raw()
    raw["project_judge_inflight"] = 2
    raw["project_judge_workers"] = 2
    assert build_config(OmegaConf.create(raw)).project_judge_workers == 2
    for invalid in (True, 0, 3, 1.5):
        raw["project_judge_workers"] = invalid
        with pytest.raises(AxBenchAdvError, match="project_judge_workers"):
            build_config(OmegaConf.create(raw))
    raw["project_judge_workers"] = 2
    raw["project_judge_inflight"] = 1
    with pytest.raises(AxBenchAdvError, match="project_judge_workers"):
        build_config(OmegaConf.create(raw))
    raw["project_judge_inflight"] = 3
    assert build_config(OmegaConf.create(raw)).project_judge_inflight == 3
    raw["project_judge_workers"] = 1
    with pytest.raises(AxBenchAdvError, match="project_judge_inflight"):
        build_config(OmegaConf.create(raw))


def test_project_reuse_single_token_mask_requires_bool() -> None:
    raw = _raw()
    raw["project_reuse_single_token_mask"] = True
    assert build_config(OmegaConf.create(raw)).project_reuse_single_token_mask is True
    for invalid in (0, 1, "true", None):
        raw["project_reuse_single_token_mask"] = invalid
        with pytest.raises(AxBenchAdvError, match="project_reuse_single_token_mask"):
            build_config(OmegaConf.create(raw))


def test_project_bf16_decoder_linears_is_counter_only_and_strict_bool() -> None:
    raw = _raw()
    raw["pressures"] = [Pressure.DIFFMEAN_COUNTER]
    raw["project_bf16_decoder_linears"] = True
    assert build_config(OmegaConf.create(raw)).project_bf16_decoder_linears is True
    for invalid in (0, 1, "true", None):
        raw["project_bf16_decoder_linears"] = invalid
        with pytest.raises(AxBenchAdvError, match="project_bf16_decoder_linears"):
            build_config(OmegaConf.create(raw))
    raw["project_bf16_decoder_linears"] = True
    raw["pressures"] = [Pressure.GEPA]
    with pytest.raises(AxBenchAdvError, match="counter-only pressures"):
        build_config(OmegaConf.create(raw))


def test_project_reuse_diffmean_texts_requires_bool() -> None:
    raw = _raw()
    assert build_config(OmegaConf.create(raw)).project_reuse_diffmean_texts is False
    raw["project_reuse_diffmean_texts"] = True
    assert build_config(OmegaConf.create(raw)).project_reuse_diffmean_texts is True
    for invalid in (0, 1, "true", None):
        raw["project_reuse_diffmean_texts"] = invalid
        with pytest.raises(AxBenchAdvError, match="project_reuse_diffmean_texts"):
            build_config(OmegaConf.create(raw))


def test_project_weight_free_extraction_requires_bool() -> None:
    raw = _raw()
    assert build_config(OmegaConf.create(raw)).project_weight_free_extraction is False
    raw["project_weight_free_extraction"] = True
    assert build_config(OmegaConf.create(raw)).project_weight_free_extraction is True
    for invalid in (0, 1, "true", None):
        raw["project_weight_free_extraction"] = invalid
        with pytest.raises(AxBenchAdvError, match="project_weight_free_extraction"):
            build_config(OmegaConf.create(raw))


def test_project_native_extraction_requires_weight_free_cuda_counter() -> None:
    raw = _raw()
    raw["pressures"] = [Pressure.DIFFMEAN_COUNTER]
    raw["project_weight_free_extraction"] = True
    raw["project_native_extraction"] = True
    assert build_config(OmegaConf.create(raw)).project_native_extraction is True
    for invalid in (0, 1, "true", None):
        raw["project_native_extraction"] = invalid
        with pytest.raises(AxBenchAdvError, match="project_native_extraction"):
            build_config(OmegaConf.create(raw))
    raw["project_native_extraction"] = True
    for change in (
        {"project_weight_free_extraction": False},
        {"device": "cpu"},
        {"pressures": [Pressure.GEPA]},
    ):
        rejected = {**raw, **change}
        with pytest.raises(AxBenchAdvError, match="weight-free CUDA counter"):
            build_config(OmegaConf.create(rejected))


def test_project_reuse_judge_workers_requires_bool() -> None:
    raw = _raw()
    assert build_config(OmegaConf.create(raw)).project_reuse_judge_workers is False
    raw["project_reuse_judge_workers"] = True
    assert build_config(OmegaConf.create(raw)).project_reuse_judge_workers is True
    for invalid in (0, 1, "true", None):
        raw["project_reuse_judge_workers"] = invalid
        with pytest.raises(AxBenchAdvError, match="project_reuse_judge_workers"):
            build_config(OmegaConf.create(raw))


def test_project_fit_selected_concepts_only_requires_bool() -> None:
    raw = _raw()
    assert build_config(OmegaConf.create(raw)).project_fit_selected_concepts_only is False
    raw["project_fit_selected_concepts_only"] = True
    assert build_config(OmegaConf.create(raw)).project_fit_selected_concepts_only is True
    for invalid in (0, 1, "true", None):
        raw["project_fit_selected_concepts_only"] = invalid
        with pytest.raises(AxBenchAdvError, match="project_fit_selected_concepts_only"):
            build_config(OmegaConf.create(raw))


def test_reuse_final_judge_worker_requires_bool() -> None:
    raw = _raw()
    raw["reuse_final_judge_worker"] = True
    assert build_config(OmegaConf.create(raw)).reuse_final_judge_worker is True
    for invalid in (0, 1, "true", None):
        raw["reuse_final_judge_worker"] = invalid
        with pytest.raises(AxBenchAdvError, match="reuse_final_judge_worker"):
            build_config(OmegaConf.create(raw))


def test_build_config_accepts_explicit_evaluation_target_receipt() -> None:
    raw = _raw()
    source = cast(dict[str, Any], raw["source"])
    source["target_receipt"] = "/source/checkpoint/evaluation_target_054600.json"

    config = build_config(OmegaConf.create(raw))

    assert config.source_paths.target_receipt == Path(
        "/source/checkpoint/evaluation_target_054600.json"
    )


def test_build_config_parses_separate_nl_attachment_without_enabling_base_nl() -> None:
    raw = _raw()
    source = cast(dict[str, Any], raw["source"])
    expectations = cast(dict[str, Any], source["expectations"])
    expectations["max_concepts"] = 50
    source["nl_attachment"] = {
        "paths": {
            "project_parquet": "/nl/project.parquet",
            "project_manifest": "/nl/run_manifest.json",
            "evaluated_jsonl": "/nl/steering.jsonl",
        },
        "expectations": {
            "max_concepts": 100,
        },
    }
    raw["factor_overrides"] = {"NLSteering": 3.0}

    config = build_config(OmegaConf.create(raw))

    assert config.source_expectations.include_nl_steering is False
    assert config.nl_attachment_paths is not None
    assert config.nl_attachment_paths.project_manifest == Path("/nl/run_manifest.json")
    assert config.nl_attachment_expectations is not None
    assert config.nl_attachment_expectations.max_concepts == 100
    assert config.factor_overrides == {"NLSteering": 3.0}


def test_worker_max_prompt_tokens_defaults_to_1024() -> None:
    parsed = build_config(OmegaConf.create(_raw()))
    config = AxBenchAdvRunConfig(
        source_paths=parsed.source_paths,
        source_expectations=parsed.source_expectations,
        source_hashes=parsed.source_hashes,
        output_dir=parsed.output_dir,
        judge_cache=parsed.judge_cache,
        reflection_cache=parsed.reflection_cache,
        reflection=parsed.reflection,
    )
    assert config.worker_max_prompt_tokens == 1024


def test_worker_max_prompt_tokens_accepts_explicit_2048() -> None:
    raw = _raw()
    raw["worker_max_prompt_tokens"] = 2048
    assert build_config(OmegaConf.create(raw)).worker_max_prompt_tokens == 2048


@pytest.mark.parametrize("value", [0, -1, 1.5, "2048", True])
def test_worker_max_prompt_tokens_rejects_non_positive_or_non_integer(value: object) -> None:
    raw = _raw()
    raw["worker_max_prompt_tokens"] = value
    with pytest.raises(AxBenchAdvError, match="worker_max_prompt_tokens"):
        build_config(OmegaConf.create(raw))


def test_build_config_reads_require_clean_worktree_false_from_expectations() -> None:
    raw = OmegaConf.create(_raw())
    raw.source.expectations.require_clean_worktree = False
    config = build_config(raw)
    assert config.source_expectations.require_clean_worktree is False


def test_worker_receives_configured_baseline_revision_and_layer() -> None:
    raw = OmegaConf.create(_raw())
    raw.source.expectations.model_revision = "it-soup-APO"
    raw.source.expectations.layer = 24
    raw.worker_max_prompt_tokens = 2048
    config = build_config(raw)
    payload = _worker_payload(config, method="DiffMean", concept_id=0, concept="test", factor=1.0)
    assert payload["model_revision"] == "it-soup-APO"
    assert payload["layer"] == 24
    assert payload["steering_layers"] == [24]
    assert payload["worker_max_prompt_tokens"] == 2048


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda raw: raw.update({"unknown": 1}), "root fields"),
        (lambda raw: raw["source"].pop("aux_parquet"), "source fields"),  # type: ignore[union-attr]
        (lambda raw: raw.update({"output_dir": "???"}), "output_dir"),
        (lambda raw: raw.update({"factor_overrides": []}), "must be a mapping"),
    ],
)
def test_build_config_fails_closed_on_representative_malformed_values(
    mutate: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    raw = _raw()
    mutate(raw)
    with pytest.raises(AxBenchAdvError, match=match):
        build_config(OmegaConf.create(raw))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("gepa_seed", 1 << 63, "gepa_seed"),
        ("reflection_minibatch_size", 11, "must not exceed"),
        ("factor_overrides", {"Unknown": 1.0}, "unknown methods"),
        ("reflection_cache", "/cache/judge.sqlite", "distinct files"),
        ("judge_cache", "/output/run/cache.sqlite", "path-disjoint"),
        ("output_dir", "/cache", "path-disjoint"),
    ],
)
def test_typed_config_preserves_relational_runtime_checks(
    field: str,
    value: object,
    match: str,
) -> None:
    raw = _raw()
    raw[field] = value
    with pytest.raises(AxBenchAdvError, match=match):
        build_config(OmegaConf.create(raw))


def test_config_parser_rejects_nonmapping_and_nontext_domains() -> None:
    raw = _raw()
    raw["source"] = []
    with pytest.raises(AxBenchAdvError, match="source fields"):
        build_config(OmegaConf.create(raw))

    raw = _raw()
    raw["reflection"] = {"wrong": True}
    with pytest.raises(AxBenchAdvError, match="reflection fields"):
        build_config(OmegaConf.create(raw))

    raw = _raw()
    source = cast(dict[str, Any], raw["source"])
    hashes = cast(dict[str, Any], source["hashes"])
    hashes["main_parquet"] = 123
    with pytest.raises(AxBenchAdvError, match="source.hashes.main_parquet"):
        build_config(OmegaConf.create(raw))


def test_hydra_main_wires_typed_config_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def fake_run(config: object) -> Path:
        seen["config"] = config
        return tmp_path / "manifest.json"

    monkeypatch.setattr(run_module, "run_pipeline", fake_run)
    run_module.main.__wrapped__(OmegaConf.create(_raw()))
    assert seen["config"].source_expectations.max_concepts == 2  # ty: ignore[unresolved-attribute]


def test_module_entrypoint_invokes_hydra_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_hydra_main(**_kwargs: object) -> Callable[[Callable[..., object]], Callable[[], None]]:
        def decorate(function: Callable[..., object]) -> Callable[[], None]:
            def invoke() -> None:
                name = getattr(function, "__name__", None)
                assert isinstance(name, str)
                calls.append(name)

            return invoke

        return decorate

    monkeypatch.setattr(hydra, "main", fake_hydra_main)
    runpy.run_path(str(Path(run_module.__file__)), run_name="__main__")
    assert calls == ["main"]


@pytest.mark.parametrize(("config_name", "max_concepts"), [("smoke", 2), ("smollm3_3b_c50", 50)])
def test_campaign_hydra_config_resolves_without_changing_the_ratified_axis(
    config_name: str, max_concepts: int
) -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs" / "eval" / "axbench_adv"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "source.prepared_dir=/source/prepared",
                "source.main_parquet=/source/main/inference/steering_data.parquet",
                "source.aux_parquet=/source/aux/inference/steering_data.parquet",
                "source.evaluated_jsonl=/source/evaluated/steering.jsonl",
                "source.project_parquet=/source/project/inference/steering_data.parquet",
                "source.target_meta=/source/checkpoint/meta_054600.json",
                "source.feature_statistics=/source/features/feature_statistics.safetensors",
                "output_dir=/output/run",
                "judge_cache=/cache/judge.sqlite",
                "reflection_cache=/cache/reflection.sqlite",
                "reflection.model=fake/reflection",
            ],
        )
    config = build_config(cfg)
    assert config.source_expectations.max_concepts == max_concepts
    assert config.source_expectations.model_revision == "it-soup-APO"
    assert dict(config.source_hashes) == {
        "main_parquet": "e72f27abed20a2fcbfdafb8e7e0055334f141c5423b09a374275d79137c996c8",
        "aux_parquet": "a76dcc6e926f2fae4310b27143811a0aa1cba9ad27ffa4a031d721b293dffc16",
        "evaluated_jsonl": "092a0ac2172a1cd32704fefddb31845ca665f3c2694beb7c3383013e7a6ebbd9",
        "project_parquet": "c6c99fbea43ca97eb4e533e858fc36e3965d3ad7aa5d3af19f97c184127bde94",
        "project_manifest": "a0823c2cfcb5692d84d4267f235854b26a041fa8ad847c9c0eafa8cf09bb3701",
        "target_receipt": "3bd98a7f73bd61d4e73b40137fb68ac9a37fa38fbd5b8a0e85a88f66f54e86f5",
    }
    assert config.source_expectations.repository_git_commit == (
        "d21c6a0b8809d2da2426e78b4b6434aa8a7057f9"
    )
    assert config.source_expectations.project_repository_git_commit == (
        "523bd1d555705c7dc812d0a7dae8b36754d245dd"
    )
    assert config.source_expectations.require_clean_worktree is False
    assert not hasattr(config, "proposal_counts")


@pytest.mark.parametrize(
    ("config_name", "target_model", "revision", "layer"),
    (
        (
            "pythsft_c50",
            "/orcd/pool/008/ychenliu/nanochat_cache/sft_baselines/pythia_sft",
            None,
            25,
        ),
        (
            "smollm3_nothink_c50",
            "HuggingFaceTB/SmolLM3-3B",
            "a07cc9a04f16550a088caea529712d1d335b0ac1",
            28,
        ),
    ),
)
def test_non_qwen_campaign_config_wires_gcg_worker_host_contract(
    config_name: str,
    target_model: str,
    revision: str | None,
    layer: int,
) -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs" / "eval" / "axbench_adv"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "source.prepared_dir=/source/prepared",
                "source.main_parquet=/source/main/inference/steering_data.parquet",
                "source.aux_parquet=/source/aux/inference/steering_data.parquet",
                "source.evaluated_jsonl=/source/evaluated/steering.jsonl",
                "source.project_parquet=/source/project/inference/steering_data.parquet",
                "source.target_meta=/source/checkpoint/meta_001125.json",
                "source.feature_statistics=/source/features/feature_statistics.safetensors",
                "output_dir=/output/run",
                "judge_cache=/cache/judge.sqlite",
                "reflection_cache=/cache/reflection.sqlite",
                "reflection.model=fake/reflection",
                "pressures=[gcg]",
                "gcg.search.optimized_token_counts=[0,1]",
                "gcg.search.max_steps=1",
            ],
        )
    config = build_config(cfg)
    payload = _worker_payload(
        config,
        method="DiffMean",
        concept_id=0,
        concept="test",
        factor=1.0,
    )
    assert tuple(config.pressures) == (Pressure.GCG,)
    assert tuple(config.gcg.search.optimized_token_counts) == (0, 1)
    assert config.source_expectations.max_concepts == 50
    assert config.worker_max_prompt_tokens == 2048
    assert payload["model_name"] == target_model
    assert payload["model_revision"] == revision
    assert payload["layer"] == layer
    assert payload["steering_layers"] == [layer]


def test_build_config_reads_alternate_project_checkpoint_pins() -> None:
    raw = OmegaConf.create(_raw())
    raw.source.expectations.project_target_step = 123
    raw.source.expectations.project_target_model_tag = "alternate-project"
    config = build_config(raw)
    assert config.source_expectations.project_target_step == 123
    assert config.source_expectations.project_target_model_tag == "alternate-project"


@pytest.mark.parametrize("value", [None, [], "not a mapping", 7])
def test_optional_nl_config_rejects_nonmapping_expectations(value: object) -> None:
    raw = OmegaConf.create(_raw())
    raw.source.expectations = value
    with pytest.raises(AxBenchAdvError, match="source.expectations fields"):
        build_config(raw)


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_nl_config_is_explicit_and_does_not_mutate_input(enabled: bool) -> None:
    raw = OmegaConf.create(_raw())
    raw.source.expectations.include_nl_steering = enabled
    raw.source.expectations.model_revision = "it-soup-APO"
    raw.source.expectations.layer = 24
    raw.factor_overrides = {"NLSteering": 3.0} if enabled else {}
    before = OmegaConf.to_container(raw, resolve=True)
    config = build_config(raw)
    assert OmegaConf.to_container(raw, resolve=True) == before
    assert config.source_expectations.include_nl_steering is enabled
    assert ("NLSteering" in config.source_expectations.methods) is enabled
    assert config.source_expectations.model_revision == "it-soup-APO"
    assert config.source_expectations.layer == 24
    assert config.factor_overrides == ({"NLSteering": 3.0} if enabled else {})


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
def test_optional_nl_config_requires_a_strict_boolean(value: object) -> None:
    raw = OmegaConf.create(_raw())
    raw.source.expectations.include_nl_steering = value
    with pytest.raises(AxBenchAdvError, match="include_nl_steering"):
        build_config(raw)


def test_pressure_hydra_fields_preserve_legacy_default_and_accept_counter_subset():
    raw = _raw()
    assert tuple(build_config(OmegaConf.create(raw)).pressures) == (Pressure.GEPA,)
    raw.update(
        pressures=[Pressure.LINEAR_PROBE_COUNTER.value],
        counter_magnitudes=[1, 2],
        project_counter_direction_root="/source/baselines",
        project_counter_layer=3,
        project_counter_fit_seed=42,
    )
    config = build_config(OmegaConf.create(raw))
    assert config.pressures == [Pressure.LINEAR_PROBE_COUNTER]
    assert config.project_counter_layer == 3
    assert config.project_counter_direction_root == Path("/source/baselines")
    assert config.project_counter_fit_seed == 42
    raw["pressures"] = ["unknown_pressure"]
    with pytest.raises(ValueError):
        build_config(OmegaConf.create(raw))
